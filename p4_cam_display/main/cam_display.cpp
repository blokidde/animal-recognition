// cam_display.cpp — ESP32-P4 + OV5647 (ESP-Video V4L2) → GC9A01 240x240 SPI LCD
// Met ESP-DL (ESP-WHO) HumanFaceDetect bounding boxes overlay.
//
// IDF 5.5.x

#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/param.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdint.h>
#include <errno.h>
#include <vector>
#include <algorithm>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"

#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "esp_assert.h"
#include "esp_system.h"

#include "driver/spi_master.h"
#include "driver/gpio.h"

#include "esp_lcd_panel_ops.h"
#include "esp_lcd_panel_io.h"
#include "esp_lcd_io_spi.h"
#include "esp_lcd_gc9a01.h"

// -------- ESP-Video / V4L2 ----------
#include "example_video_common.h"
#include <linux/videodev2.h>

// -------- ESP-DL / ESP-WHO (face detect) ----------
#include "dl_image.hpp"          // img_t + utils
#include "human_face_detect.hpp" // HumanFaceDetect (MSR01-achtig)

// logging tag
static const char *TAG = "P4_CAM_GC9A01_MIN_FAST_SAFE";

// ===== LCD & pins =====
static constexpr int LCD_W = 240;
static constexpr int LCD_H = 240;

static constexpr spi_host_device_t LCD_HOST = SPI2_HOST;
static constexpr int PIN_LCD_SCLK = 32;
static constexpr int PIN_LCD_MOSI = 26;
static constexpr int PIN_LCD_MISO = -1; // write-only
static constexpr int PIN_LCD_CS = 25;
static constexpr int PIN_LCD_DC = 27;
static constexpr int PIN_LCD_RST = 20;
static constexpr int PIN_LCD_BL = 21;

static constexpr uint32_t LCD_SPI_HZ = (80u * 1000u * 1000u);

// FILL-crop fine-tuning
static constexpr int CALIB_SHIFT_X = 300;
static constexpr int CALIB_SHIFT_Y = 30;
static constexpr float EXTRA_ZOOM = 0.04f;

// ---- GC9A01 command set ----
#define GC9A01_CMD_CASET 0x2A
#define GC9A01_CMD_RASET 0x2B
#define GC9A01_CMD_RAMWR 0x2C

// ---- small helpers ----
static inline gpio_num_t to_gpio(int pin) { return (pin >= 0) ? (gpio_num_t)pin : GPIO_NUM_NC; }
static inline uint16_t bswap16(uint16_t v) { return (uint16_t)((v << 8) | (v >> 8)); }
static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

// ---------- LCD transfer completion sync ----------
static SemaphoreHandle_t g_lcd_done = nullptr;
static bool on_color_trans_done_cb(esp_lcd_panel_io_handle_t,
                                   esp_lcd_panel_io_event_data_t *,
                                   void *)
{
    BaseType_t hpw = pdFALSE;
    if (g_lcd_done)
        xSemaphoreGiveFromISR(g_lcd_done, &hpw);
    return hpw == pdTRUE;
}

static void requeue_camera_buffer(int fd, struct v4l2_buffer *buf)
{
    if (ioctl(fd, VIDIOC_QBUF, buf) != 0)
    {
        ESP_LOGW(TAG, "QBUF fail: %d", errno);
    }
}

static bool dequeue_latest_frame(int fd, struct v4l2_buffer *out)
{
    bool have_frame = false;

    while (true)
    {
        struct v4l2_buffer candidate{};
        candidate.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        candidate.memory = V4L2_MEMORY_MMAP;

        if (ioctl(fd, VIDIOC_DQBUF, &candidate) != 0)
        {
            if (errno == EAGAIN || errno == EWOULDBLOCK)
            {
                if (have_frame)
                    return true;

                vTaskDelay(1);
                continue;
            }

            ESP_LOGW(TAG, "DQBUF fail: %d", errno);
            vTaskDelay(1);
            return false;
        }

        if (!(candidate.flags & V4L2_BUF_FLAG_DONE))
        {
            requeue_camera_buffer(fd, &candidate);
            continue;
        }

        if (have_frame)
            requeue_camera_buffer(fd, out);

        *out = candidate;
        have_frame = true;
    }
}

// ---------- Camera open helper ----------
static int open_first_available_cam(char out_path[32])
{
    const char *try_list[] = {
        "/dev/mipi_csi_cam0", "/dev/dvp_cam0", "/dev/spi_cam0", "/dev/uvc0", "/dev/video0"};
    for (size_t i = 0; i < sizeof(try_list) / sizeof(try_list[0]); ++i)
    {
        int fd = open(try_list[i], O_RDWR | O_NONBLOCK);
        if (fd >= 0)
        {
            strncpy(out_path, try_list[i], 31);
            out_path[31] = '\0';
            return fd;
        }
    }
    out_path[0] = '\0';
    return -1;
}

// Build mapping: dst i in [0..outN-1] → src_start + floor((i*(srcN-1))/(outN-1))
static void build_map_q16(int *map, int outN, int src_start, int srcN)
{
    if (outN <= 1)
    {
        map[0] = src_start;
        return;
    }
    const uint32_t inc = ((uint32_t)(srcN - 1) << 16) / (uint32_t)(outN - 1);
    uint32_t acc = 0;
    for (int i = 0; i < outN; ++i)
    {
        map[i] = src_start + (int)(acc >> 16);
        acc += inc;
    }
}

// ---- RGB helpers for overlays ----
static inline uint16_t rgb565_swapped(uint8_t r, uint8_t g, uint8_t b)
{
    uint16_t c = (uint16_t)(((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3));
    return bswap16(c);
}
static inline void fill_run16(uint16_t *p, uint16_t v, int n)
{
    for (int i = 0; i < n; i++)
        p[i] = v;
}
static void draw_rect_rgb565(uint16_t *fb, int W, int H,
                             int x1, int y1, int x2, int y2,
                             uint16_t col, int thickness = 2)
{
    x1 = clampi(x1, 0, W - 1);
    y1 = clampi(y1, 0, H - 1);
    x2 = clampi(x2, 0, W - 1);
    y2 = clampi(y2, 0, H - 1);
    if (x2 < x1)
    {
        int t = x1;
        x1 = x2;
        x2 = t;
    }
    if (y2 < y1)
    {
        int t = y1;
        y1 = y2;
        y2 = t;
    }

    // horizontale lijnen
    for (int t = 0; t < thickness; ++t)
    {
        int ytop = y1 + t;
        if (ytop <= y2)
            fill_run16(fb + ytop * W + x1, col, (x2 - x1 + 1));
        int ybot = y2 - t;
        if (ybot >= y1)
            fill_run16(fb + ybot * W + x1, col, (x2 - x1 + 1));
    }
    // verticale lijnen
    for (int y = y1; y <= y2; ++y)
    {
        for (int t = 0; t < thickness; ++t)
        {
            int xl = x1 + t;
            if (xl <= x2)
                fb[y * W + xl] = col;
            int xr = x2 - t;
            if (xr >= x1)
                fb[y * W + xr] = col;
        }
    }
}

// ---- Downscale RGB565 (240x240) → RGB888 (detW x detH) ----
static void downscale_rgb565_to_rgb888(const uint16_t *src_rgb565, int srcW, int srcH,
                                       uint8_t *dst_rgb888, int detW, int detH)
{
    static int xmap[320];
    static int ymap[320];
    static int cached_srcW = -1;
    static int cached_srcH = -1;
    static int cached_detW = -1;
    static int cached_detH = -1;

    if (cached_srcW != srcW || cached_srcH != srcH || cached_detW != detW || cached_detH != detH)
    {
        build_map_q16(xmap, detW, 0, srcW);
        build_map_q16(ymap, detH, 0, srcH);
        cached_srcW = srcW;
        cached_srcH = srcH;
        cached_detW = detW;
        cached_detH = detH;
    }

    int di = 0;
    for (int dy = 0; dy < detH; ++dy)
    {
        const uint16_t *srow = src_rgb565 + (size_t)ymap[dy] * srcW;
        for (int dx = 0; dx < detW; ++dx)
        {
            uint16_t p = bswap16(srow[xmap[dx]]);
            int r = (p >> 11) & 0x1F;
            r = (r * 255 + 15) / 31;
            int g = (p >> 5) & 0x3F;
            g = (g * 255 + 31) / 63;
            int b = p & 0x1F;
            b = (b * 255 + 15) / 31;
            dst_rgb888[di++] = (uint8_t)r;
            dst_rgb888[di++] = (uint8_t)g;
            dst_rgb888[di++] = (uint8_t)b;
        }
    }
}

extern "C" void app_main(void)
{
    // 1) Init ESP-Video (V4L2)
    ESP_ERROR_CHECK(example_video_init());
    ESP_LOGI(TAG, "ESP-Video initialized");

    // 2) Open camera
    char used_dev[32];
    int fd = open_first_available_cam(used_dev);
    if (fd < 0)
    {
        ESP_LOGE(TAG, "Geen /dev/videoX device gevonden.");
        vTaskDelay(portMAX_DELAY);
    }
    ESP_LOGI(TAG, "Video device: %s", used_dev);

    // 3) Force RGB565 @ 800x640
    struct v4l2_format set_fmt{};
    set_fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    set_fmt.fmt.pix.width = 800;
    set_fmt.fmt.pix.height = 640;
    set_fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB565;
    set_fmt.fmt.pix.field = V4L2_FIELD_NONE;
    (void)ioctl(fd, VIDIOC_S_FMT, &set_fmt);

    struct v4l2_format fmt{};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd, VIDIOC_G_FMT, &fmt) != 0)
    {
        ESP_LOGE(TAG, "VIDIOC_G_FMT fail: %d", errno);
        vTaskDelay(portMAX_DELAY);
    }

    if (fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_RGB565)
    {
        ESP_LOGE(TAG, "RGB565 verwacht. FourCC=0x%08X", fmt.fmt.pix.pixelformat);
        vTaskDelay(portMAX_DELAY);
    }
    const int src_w = (int)fmt.fmt.pix.width;
    const int src_h = (int)fmt.fmt.pix.height;
    int stride_bytes = (int)fmt.fmt.pix.bytesperline;
    if (stride_bytes <= 0)
        stride_bytes = src_w * 2;

    ESP_LOGI(TAG, "Camera fmt: %dx%d RGB565, stride=%d", src_w, src_h, stride_bytes);

    // 4) MMAP buffers + STREAMON
    struct v4l2_requestbuffers req{};
    req.count = 2;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (ioctl(fd, VIDIOC_REQBUFS, &req) != 0 || req.count < 2)
    {
        ESP_LOGE(TAG, "REQBUFS fail (count=%u)", req.count);
        vTaskDelay(portMAX_DELAY);
    }

    struct MMapBuf
    {
        void *start;
        size_t length;
    } buffers[8]{};
    for (uint32_t i = 0; i < req.count; ++i)
    {
        struct v4l2_buffer b{};
        b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        b.memory = V4L2_MEMORY_MMAP;
        b.index = i;
        if (ioctl(fd, VIDIOC_QUERYBUF, &b) != 0)
        {
            ESP_LOGE(TAG, "QUERYBUF %u fail: %d", i, errno);
            vTaskDelay(portMAX_DELAY);
        }
        buffers[i].length = b.length;
        buffers[i].start = mmap(NULL, b.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, b.m.offset);
        if (buffers[i].start == MAP_FAILED)
        {
            ESP_LOGE(TAG, "mmap %u fail", i);
            vTaskDelay(portMAX_DELAY);
        }
        if (ioctl(fd, VIDIOC_QBUF, &b) != 0)
        {
            ESP_LOGE(TAG, "QBUF %u fail: %d", i, errno);
            vTaskDelay(portMAX_DELAY);
        }
    }
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd, VIDIOC_STREAMON, &type) != 0)
    {
        ESP_LOGE(TAG, "STREAMON fail: %d", errno);
        vTaskDelay(portMAX_DELAY);
    }

    // 5) LCD init
    spi_bus_config_t buscfg{};
    buscfg.sclk_io_num = PIN_LCD_SCLK;
    buscfg.mosi_io_num = PIN_LCD_MOSI;
    buscfg.miso_io_num = PIN_LCD_MISO;
    buscfg.quadwp_io_num = -1;
    buscfg.quadhd_io_num = -1;
    buscfg.max_transfer_sz = LCD_W * LCD_H * (int)sizeof(uint16_t) + 64;
    ESP_ERROR_CHECK(spi_bus_initialize(LCD_HOST, &buscfg, SPI_DMA_CH_AUTO));

    if (PIN_LCD_BL >= 0)
    {
        gpio_config_t io{};
        io.pin_bit_mask = (1ULL << (unsigned)PIN_LCD_BL);
        io.mode = GPIO_MODE_OUTPUT;
        gpio_config(&io);
        gpio_set_level(to_gpio(PIN_LCD_BL), 1);
    }
    if (PIN_LCD_SCLK >= 0)
        gpio_set_drive_capability(to_gpio(PIN_LCD_SCLK), GPIO_DRIVE_CAP_3);
    if (PIN_LCD_MOSI >= 0)
        gpio_set_drive_capability(to_gpio(PIN_LCD_MOSI), GPIO_DRIVE_CAP_3);
    if (PIN_LCD_DC >= 0)
        gpio_set_drive_capability(to_gpio(PIN_LCD_DC), GPIO_DRIVE_CAP_3);
    if (PIN_LCD_CS >= 0)
        gpio_set_drive_capability(to_gpio(PIN_LCD_CS), GPIO_DRIVE_CAP_3);

    g_lcd_done = xSemaphoreCreateBinary();
    configASSERT(g_lcd_done);

    esp_lcd_panel_io_handle_t io_handle = nullptr;
    esp_lcd_panel_io_spi_config_t io_cfg{};
    io_cfg.dc_gpio_num = to_gpio(PIN_LCD_DC);
    io_cfg.cs_gpio_num = to_gpio(PIN_LCD_CS);
    io_cfg.pclk_hz = LCD_SPI_HZ;
    io_cfg.lcd_cmd_bits = 8;
    io_cfg.lcd_param_bits = 8;
    io_cfg.spi_mode = 0;
    io_cfg.trans_queue_depth = 1;
    io_cfg.on_color_trans_done = on_color_trans_done_cb;
    ESP_ERROR_CHECK(esp_lcd_new_panel_io_spi((esp_lcd_spi_bus_handle_t)LCD_HOST, &io_cfg, &io_handle));

    esp_lcd_panel_handle_t panel = nullptr;
    esp_lcd_panel_dev_config_t panel_cfg{};
    panel_cfg.reset_gpio_num = to_gpio(PIN_LCD_RST);
    panel_cfg.rgb_ele_order = LCD_RGB_ELEMENT_ORDER_BGR;
    panel_cfg.bits_per_pixel = 16;
    ESP_ERROR_CHECK(esp_lcd_new_panel_gc9a01(io_handle, &panel_cfg, &panel));

    ESP_ERROR_CHECK(esp_lcd_panel_reset(panel));
    vTaskDelay(pdMS_TO_TICKS(20));
    ESP_ERROR_CHECK(esp_lcd_panel_init(panel));
    vTaskDelay(pdMS_TO_TICKS(120));
    ESP_ERROR_CHECK(esp_lcd_panel_disp_on_off(panel, true));
    ESP_ERROR_CHECK(esp_lcd_panel_invert_color(panel, true));
    ESP_ERROR_CHECK(esp_lcd_panel_swap_xy(panel, false));
    ESP_ERROR_CHECK(esp_lcd_panel_mirror(panel, false, false));
    ESP_ERROR_CHECK(esp_lcd_panel_set_gap(panel, 0, 0));

    auto set_full_window = [&](void)
    {
        uint8_t caset[4] = {0x00, 0x00, 0x00, (uint8_t)(LCD_W - 1)};
        uint8_t raset[4] = {0x00, 0x00, 0x00, (uint8_t)(LCD_H - 1)};
        ESP_ERROR_CHECK(esp_lcd_panel_io_tx_param(io_handle, GC9A01_CMD_CASET, caset, sizeof(caset)));
        ESP_ERROR_CHECK(esp_lcd_panel_io_tx_param(io_handle, GC9A01_CMD_RASET, raset, sizeof(raset)));
    };

    // 6) Twee DMA-capable framebuffers
    const size_t FB_PIXELS = (size_t)LCD_W * LCD_H;
    const size_t FB_BYTES = FB_PIXELS * sizeof(uint16_t);
    uint16_t *fb[2] = {nullptr, nullptr};
    for (int i = 0; i < 2; ++i)
    {
        fb[i] = (uint16_t *)heap_caps_malloc(FB_BYTES, MALLOC_CAP_DMA | MALLOC_CAP_INTERNAL);
        configASSERT(fb[i] && "kon geen INTERNAL DMA buffer allocaten");
    }
    int cur = 0;

    // 7) ROI voor FILL
    int base_sq = (src_w < src_h) ? src_w : src_h;
    int extra = (int)((float)base_sq * EXTRA_ZOOM + 0.5f);
    int roi_size = base_sq - extra;
    if (roi_size < 16)
        roi_size = 16;

    int x0 = (src_w - roi_size) / 2;
    int y0 = (src_h - roi_size) / 2;
    x0 = clampi(x0 + CALIB_SHIFT_X, 0, src_w - roi_size);
    y0 = clampi(y0 + CALIB_SHIFT_Y, 0, src_h - roi_size);

    const int roi_w = roi_size;
    const int roi_h = roi_size;

    ESP_LOGI(TAG, "FILL ROI=%d @(%d,%d) from %dx%d (zoom≈%.1f%%)",
             roi_size, x0, y0, src_w, src_h, 100.0f * (float)extra / (float)base_sq);

    static int xmap[LCD_W], ymap[LCD_H];
    build_map_q16(xmap, LCD_W, x0, roi_w);
    build_map_q16(ymap, LCD_H, y0, roi_h);

    // ===== Face detector: init & buffers =====
    HumanFaceDetect face_det;     // uit human_face_detect.hpp
    face_det.set_score_thr(0.5f); // drempel
    face_det.set_nms_thr(0.3f);   // IoU NMS

    // detecteren op 160x120 RGB888
    static constexpr int DET_W = 160;
    static constexpr int DET_H = 120;
    uint8_t *det_rgb = (uint8_t *)heap_caps_malloc(DET_W * DET_H * 3, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    configASSERT(det_rgb);

    // cache
    std::vector<dl::detect::result_t> last_faces;
    const int DETECT_EVERY_N = 5;
    int detect_countdown = 0;
    int yield_countdown = 30;
    last_faces.reserve(8);

    // ===== FPS & timings =====
    uint64_t t_fps0 = esp_timer_get_time();
    int fps_frames = 0;
    double avg_scale_ms = 0.0, avg_submit_ms = 0.0;
    const double alpha = 0.2;

    bool dma_active = false;

    // 8) Main loop
    while (true)
    {
        // Dequeue een frame
        struct v4l2_buffer buf{};
        if (!dequeue_latest_frame(fd, &buf))
            continue;

        const uint8_t *frame = (const uint8_t *)buffers[buf.index].start;
        uint16_t *dst = fb[cur];

        // === Scale: ROI (RGB565) -> 240x240 ===
        uint64_t t0 = esp_timer_get_time();
        for (int dy = 0; dy < LCD_H; ++dy)
        {
            const int sy = ymap[dy];
            const uint16_t *src_line = (const uint16_t *)(frame + (size_t)sy * (size_t)stride_bytes);
            uint16_t *out_line = dst + (size_t)dy * LCD_W;

            int dx = 0;
            for (; dx <= LCD_W - 8; dx += 8)
            {
                out_line[dx + 0] = bswap16(src_line[xmap[dx + 0]]);
                out_line[dx + 1] = bswap16(src_line[xmap[dx + 1]]);
                out_line[dx + 2] = bswap16(src_line[xmap[dx + 2]]);
                out_line[dx + 3] = bswap16(src_line[xmap[dx + 3]]);
                out_line[dx + 4] = bswap16(src_line[xmap[dx + 4]]);
                out_line[dx + 5] = bswap16(src_line[xmap[dx + 5]]);
                out_line[dx + 6] = bswap16(src_line[xmap[dx + 6]]);
                out_line[dx + 7] = bswap16(src_line[xmap[dx + 7]]);
            }
            for (; dx < LCD_W; ++dx)
            {
                out_line[dx] = bswap16(src_line[xmap[dx]]);
            }
        }
        uint64_t t1 = esp_timer_get_time();

        // === Face detect 1x per N frames ===
        if (detect_countdown <= 0)
        {
            detect_countdown = DETECT_EVERY_N;

            // 240x240 RGB565 (dst) → 160x120 RGB888 (det_rgb)
            downscale_rgb565_to_rgb888(dst, LCD_W, LCD_H, det_rgb, DET_W, DET_H);

            // dl::image::img_t vullen — alleen data/width/height bestaan in jouw versie
            dl::image::img_t det_img{};
            det_img.data = det_rgb;
            det_img.width = DET_W;
            det_img.height = DET_H;
            // geen det_img.channel / det_img.stride zetten!

            // run detector (HumanFaceDetect::run) → std::list<dl::detect::result_t>
            auto faces_list = face_det.run(det_img);
            last_faces.assign(faces_list.begin(), faces_list.end()); // list → vector cache
        }
        --detect_countdown;

        // === Boxes tekenen op 240x240 ===
        const uint16_t COL_BOX = rgb565_swapped(255, 32, 32); // rood
        for (const auto &f : last_faces)
        {
            // f.box = [x, y, w, h] op DET_W x DET_H
            int x = (int)f.box[0];
            int y = (int)f.box[1];
            int w = (int)f.box[2];
            int h = (int)f.box[3];

            int x0b = (x * LCD_W) / DET_W;
            int y0b = (y * LCD_H) / DET_H;
            int x1b = ((x + w) * LCD_W) / DET_W;
            int y1b = ((y + h) * LCD_H) / DET_H;

            draw_rect_rgb565(dst, LCD_W, LCD_H, x0b, y0b, x1b, y1b, COL_BOX, 2);
        }

        // === DMA submit ===
        if (dma_active)
        {
            if (xSemaphoreTake(g_lcd_done, pdMS_TO_TICKS(100)) != pdTRUE)
            {
                ESP_LOGW(TAG, "LCD DMA timeout; frame skipped");
                requeue_camera_buffer(fd, &buf);
                continue;
            }
            dma_active = false;
        }
        set_full_window();
        ESP_ERROR_CHECK(esp_lcd_panel_io_tx_color(io_handle, GC9A01_CMD_RAMWR, dst, FB_BYTES));
        dma_active = true;

        uint64_t t2 = esp_timer_get_time();

        // buffer terug naar driver
        requeue_camera_buffer(fd, &buf);

        // timings / FPS
        double scale_ms = (t1 - t0) / 1000.0;
        double submit_ms = (t2 - t1) / 1000.0;
        avg_scale_ms = (avg_scale_ms == 0.0) ? scale_ms : (alpha * scale_ms + (1.0 - alpha) * avg_scale_ms);
        avg_submit_ms = (avg_submit_ms == 0.0) ? submit_ms : (alpha * submit_ms + (1.0 - alpha) * avg_submit_ms);

        fps_frames++;
        uint64_t now = t2;
        if (now - t_fps0 >= 1000000ULL)
        {
            double secs = (now - t_fps0) / 1000000.0;
            double fps = fps_frames / secs;
            ESP_LOGI(TAG, "FPS: %.1f | scale: %.2f ms | submit: %.2f ms (N=%d)",
                     fps, avg_scale_ms, avg_submit_ms, DETECT_EVERY_N);
            fps_frames = 0;
            t_fps0 = now;
        }

        if (--yield_countdown <= 0)
        {
            yield_countdown = 30;
            vTaskDelay(1);
        }

        cur ^= 1;
    }
}
