// cam_display_minimal.cpp — ESP32-P4 + OV5647 (ESP-Video) → GC9A01 240x240 SPI LCD
// Minimal, stabiel: RGB565-only, FILL-only, DMA-sync, 60 MHz SPI, geen onnodige branches.

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

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"

#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "esp_assert.h"
#include "esp_system.h"
#include "esp_mac.h"

#include "driver/spi_master.h"
#include "driver/gpio.h"

#include "esp_lcd_panel_ops.h"
#include "esp_lcd_panel_io.h"
#include "esp_lcd_io_spi.h"
#include "esp_lcd_gc9a01.h"

// -------- ESP-Video / V4L2 ----------
#include "example_video_common.h"
#include <linux/videodev2.h>

static const char* TAG = "P4_CAM_GC9A01_MIN";

// ===== Hard settings =====
static constexpr int LCD_W = 240;
static constexpr int LCD_H = 240;

static constexpr spi_host_device_t LCD_HOST = SPI2_HOST;
static constexpr int PIN_LCD_SCLK = 32;
static constexpr int PIN_LCD_MOSI = 26;
static constexpr int PIN_LCD_MISO = -1; // write-only
static constexpr int PIN_LCD_CS   = 25;
static constexpr int PIN_LCD_DC   = 27;
static constexpr int PIN_LCD_RST  = 20; // -1 als geen RST
static constexpr int PIN_LCD_BL   = 21; // -1 als geen backlight

// Iets conservatiever dan 80 MHz
static constexpr uint32_t LCD_SPI_HZ = (60u * 1000u * 1000u);

// software FILL-crop finetune (pixels in bronbeeld)
static constexpr int CALIB_SHIFT_X = 300;   // positief => rechts
static constexpr int CALIB_SHIFT_Y = 30;    // positief => beneden
static constexpr float EXTRA_ZOOM  = 0.04f; // 4% extra; 0.00..0.08 aanhouden

// ---- helpers ----
static inline gpio_num_t to_gpio(int pin) { return (pin >= 0) ? static_cast<gpio_num_t>(pin) : GPIO_NUM_NC; }
static inline uint16_t bswap16(uint16_t v) { return (uint16_t)((v << 8) | (v >> 8)); }
static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
static inline int lerp_idx_full(int d, int d_out, int r_in) {
    if (d_out <= 1) return 0;
    return (int)((int64_t)d * (r_in - 1) / (d_out - 1));
}

// ---------- LCD transfer-done sync ----------
static SemaphoreHandle_t g_lcd_done = nullptr;
static bool g_lcd_started = false;

static bool on_color_trans_done_cb(esp_lcd_panel_io_handle_t /*io*/,
                                   esp_lcd_panel_io_event_data_t* /*edata*/,
                                   void* /*user_ctx*/)
{
    BaseType_t hpw = pdFALSE;
    if (g_lcd_done) xSemaphoreGiveFromISR(g_lcd_done, &hpw);
    return hpw == pdTRUE;
}

// ---------- Camera open helper ----------
static int open_first_available_cam(char out_path[32])
{
    const char* try_list[] = {
        "/dev/mipi_csi_cam0", "/dev/dvp_cam0", "/dev/spi_cam0", "/dev/uvc0", "/dev/video0"
    };
    for (size_t i=0; i<sizeof(try_list)/sizeof(try_list[0]); ++i) {
        int fd = open(try_list[i], O_RDWR);
        if (fd >= 0) {
            strncpy(out_path, try_list[i], 31);
            out_path[31] = '\0';
            return fd;
        }
    }
    out_path[0] = '\0';
    return -1;
}

extern "C" void app_main(void)
{
    // 1) Init video
    ESP_ERROR_CHECK(example_video_init());
    ESP_LOGI(TAG, "ESP-Video initialized");

    // 2) Open cam
    char used_dev[32];
    int fd = open_first_available_cam(used_dev);
    if (fd < 0) {
        ESP_LOGE(TAG, "Geen /dev/videoX device gevonden.");
        vTaskDelay(portMAX_DELAY);
    }
    ESP_LOGI(TAG, "Video device: %s", used_dev);

    // 3) Dwing RGB565 aan (driver mag height/width bijstellen; dat is ok)
    struct v4l2_format set_fmt{};
    set_fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    set_fmt.fmt.pix.width        = 800;
    set_fmt.fmt.pix.height       = 640;
    set_fmt.fmt.pix.pixelformat  = V4L2_PIX_FMT_RGB565;
    set_fmt.fmt.pix.field        = V4L2_FIELD_NONE;
    (void)ioctl(fd, VIDIOC_S_FMT, &set_fmt);

    struct v4l2_format fmt{};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd, VIDIOC_G_FMT, &fmt) != 0) {
        ESP_LOGE(TAG, "VIDIOC_G_FMT faalde: %d", errno);
        vTaskDelay(portMAX_DELAY);
    }

    const uint32_t fourcc = fmt.fmt.pix.pixelformat;
    if (fourcc != V4L2_PIX_FMT_RGB565) {
        ESP_LOGE(TAG, "Deze minimal build verwacht RGB565. FourCC=0x%08X", fourcc);
        vTaskDelay(portMAX_DELAY);
    }

    const int src_w = (int)fmt.fmt.pix.width;
    const int src_h = (int)fmt.fmt.pix.height;
    int stride_bytes = (int)fmt.fmt.pix.bytesperline;
    if (stride_bytes <= 0) stride_bytes = src_w * 2;

    ESP_LOGI(TAG, "Camera fmt: %dx%d RGB565, stride=%d bytes", src_w, src_h, stride_bytes);

    // 4) Buffers & stream
    struct v4l2_requestbuffers req{};
    req.count  = 4;
    req.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (ioctl(fd, VIDIOC_REQBUFS, &req) != 0 || req.count < 2) {
        ESP_LOGE(TAG, "REQBUFS fail / te weinig buffers (count=%u)", req.count);
        vTaskDelay(portMAX_DELAY);
    }

    struct MMapBuf { void* start; size_t length; } buffers[8] = {};
    for (uint32_t i=0; i<req.count; ++i) {
        struct v4l2_buffer b{};
        b.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        b.memory = V4L2_MEMORY_MMAP;
        b.index  = i;
        if (ioctl(fd, VIDIOC_QUERYBUF, &b) != 0) {
            ESP_LOGE(TAG, "QUERYBUF %u faalde: %d", i, errno);
            vTaskDelay(portMAX_DELAY);
        }
        buffers[i].length = b.length;
        buffers[i].start  = mmap(NULL, b.length, PROT_READ|PROT_WRITE, MAP_SHARED, fd, b.m.offset);
        if (buffers[i].start == MAP_FAILED) {
            ESP_LOGE(TAG, "mmap failed buffer %u", i);
            vTaskDelay(portMAX_DELAY);
        }
        if (ioctl(fd, VIDIOC_QBUF, &b) != 0) {
            ESP_LOGE(TAG, "QBUF %u faalde: %d", i, errno);
            vTaskDelay(portMAX_DELAY);
        }
    }
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd, VIDIOC_STREAMON, &type) != 0) {
        ESP_LOGE(TAG, "STREAMON faalde: %d", errno);
        vTaskDelay(portMAX_DELAY);
    }

    // 5) LCD init (queue depth = 1) + DMA sync
    spi_bus_config_t buscfg{};
    buscfg.sclk_io_num     = PIN_LCD_SCLK;
    buscfg.mosi_io_num     = PIN_LCD_MOSI;
    buscfg.miso_io_num     = PIN_LCD_MISO;
    buscfg.quadwp_io_num   = -1;
    buscfg.quadhd_io_num   = -1;
    buscfg.max_transfer_sz = LCD_W * LCD_H * (int)sizeof(uint16_t) + 16;
    ESP_ERROR_CHECK(spi_bus_initialize(LCD_HOST, &buscfg, SPI_DMA_CH_AUTO));

    // Backlight aan
    if (PIN_LCD_BL >= 0) {
        gpio_config_t io_conf{};
        io_conf.pin_bit_mask = (1ULL << (unsigned)PIN_LCD_BL);
        io_conf.mode = GPIO_MODE_OUTPUT;
        gpio_config(&io_conf);
        gpio_set_level(to_gpio(PIN_LCD_BL), 1);
    }

    // Sterkere drive op snelle lijnen
    if (PIN_LCD_SCLK >= 0) gpio_set_drive_capability(to_gpio(PIN_LCD_SCLK), GPIO_DRIVE_CAP_3);
    if (PIN_LCD_MOSI >= 0) gpio_set_drive_capability(to_gpio(PIN_LCD_MOSI), GPIO_DRIVE_CAP_3);
    if (PIN_LCD_DC   >= 0) gpio_set_drive_capability(to_gpio(PIN_LCD_DC),   GPIO_DRIVE_CAP_3);
    if (PIN_LCD_CS   >= 0) gpio_set_drive_capability(to_gpio(PIN_LCD_CS),   GPIO_DRIVE_CAP_3);

    g_lcd_done = xSemaphoreCreateBinary();
    configASSERT(g_lcd_done);

    esp_lcd_panel_io_handle_t io_handle = nullptr;
    esp_lcd_panel_io_spi_config_t io_cfg{};
    io_cfg.dc_gpio_num          = to_gpio(PIN_LCD_DC);
    io_cfg.cs_gpio_num          = to_gpio(PIN_LCD_CS);
    io_cfg.pclk_hz              = LCD_SPI_HZ;
    io_cfg.lcd_cmd_bits         = 8;
    io_cfg.lcd_param_bits       = 8;
    io_cfg.spi_mode             = 0;              // indien flits blijft: test 3
    io_cfg.trans_queue_depth    = 1;              // 1 in-flight transfer
    io_cfg.on_color_trans_done  = on_color_trans_done_cb;
    ESP_ERROR_CHECK(esp_lcd_new_panel_io_spi((esp_lcd_spi_bus_handle_t)LCD_HOST, &io_cfg, &io_handle));

    esp_lcd_panel_handle_t panel = nullptr;
    esp_lcd_panel_dev_config_t panel_cfg{};
    panel_cfg.reset_gpio_num = to_gpio(PIN_LCD_RST);
    panel_cfg.rgb_ele_order  = LCD_RGB_ELEMENT_ORDER_BGR;  // vaak BGR op GC9A01-modules
    panel_cfg.bits_per_pixel = 16;                         // RGB565
    ESP_ERROR_CHECK(esp_lcd_new_panel_gc9a01(io_handle, &panel_cfg, &panel));

    ESP_ERROR_CHECK(esp_lcd_panel_reset(panel));
    vTaskDelay(pdMS_TO_TICKS(20));
    ESP_ERROR_CHECK(esp_lcd_panel_init(panel));
    vTaskDelay(pdMS_TO_TICKS(120));
    ESP_ERROR_CHECK(esp_lcd_panel_disp_on_off(panel, true));
    ESP_ERROR_CHECK(esp_lcd_panel_invert_color(panel, true)); // ← meestal rustiger dan true
    ESP_ERROR_CHECK(esp_lcd_panel_swap_xy(panel, false));
    ESP_ERROR_CHECK(esp_lcd_panel_mirror(panel, false, false));
    ESP_ERROR_CHECK(esp_lcd_panel_set_gap(panel, 0, 0));

    // 6) Twee framebuffers in INTERNAL RAM
    const size_t FB_PIXELS = (size_t)LCD_W * LCD_H;
    const size_t FB_BYTES  = FB_PIXELS * sizeof(uint16_t);
    uint16_t* fb[2] = {nullptr, nullptr};
    for (int i=0; i<2; ++i) {
        fb[i] = (uint16_t*) heap_caps_malloc(FB_BYTES, MALLOC_CAP_DMA | MALLOC_CAP_INTERNAL);
        configASSERT(fb[i] && "kon geen INTERNAL DMA-cap buffer allocaten");
    }
    int cur = 0;

    // 7) Software center-crop ROI (vierkant) + klein beetje extra zoom
    int base_sq  = (src_w < src_h) ? src_w : src_h;
    int extra    = (int)((float)base_sq * EXTRA_ZOOM + 0.5f);
    int roi_size = base_sq - extra; if (roi_size < 16) roi_size = 16;

    int x0 = (src_w - roi_size) / 2;
    int y0 = (src_h - roi_size) / 2;

    x0 = clampi(x0 + CALIB_SHIFT_X, 0, src_w - roi_size);
    y0 = clampi(y0 + CALIB_SHIFT_Y, 0, src_h - roi_size);

    const int roi_w = roi_size;
    const int roi_h = roi_size;

    ESP_LOGI(TAG, "FILL ROI=%d @(%d,%d) from %dx%d (zoom≈%.1f%%)", roi_size, x0, y0, src_w, src_h,
             100.0f * (float)extra / (float)base_sq);

    // 8) Capture + scale + push
    while (true) {
        // Dequeue
        struct v4l2_buffer buf{};
        buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        if (ioctl(fd, VIDIOC_DQBUF, &buf) != 0) {
            ESP_LOGW(TAG, "DQBUF fail: %d", errno);
            vTaskDelay(pdMS_TO_TICKS(1));
            continue;
        }
        if (!(buf.flags & V4L2_BUF_FLAG_DONE)) {
            (void)ioctl(fd, VIDIOC_QBUF, &buf);
            continue;
        }

        const uint8_t* frame = (const uint8_t*)buffers[buf.index].start;
        uint16_t* dst = fb[cur];

        // Schalen RGB565 (nearest) van ROI → 240x240
        for (int dy=0; dy<LCD_H; ++dy) {
            int sy = y0 + lerp_idx_full(dy, LCD_H, roi_h);
            const uint16_t* src_line = (const uint16_t*)(frame + (size_t)sy * (size_t)stride_bytes);
            uint16_t* out_line = dst + (size_t)dy * LCD_W;

            for (int dx=0; dx<LCD_W; ++dx) {
                int sx = x0 + lerp_idx_full(dx, LCD_W, roi_w);
                out_line[dx] = bswap16(src_line[sx]); // LCD verwacht MSB-first
            }
        }

        // Push + sync
        ESP_ERROR_CHECK(esp_lcd_panel_draw_bitmap(panel, 0, 0, LCD_W, LCD_H, dst));
        // wacht tot DMA klaar is
        if (!g_lcd_started) {
            xSemaphoreTake(g_lcd_done, pdMS_TO_TICKS(50));
            g_lcd_started = true;
        } else {
            xSemaphoreTake(g_lcd_done, pdMS_TO_TICKS(50));
        }

        // Requeue capture buffer
        if (ioctl(fd, VIDIOC_QBUF, &buf) != 0) {
            ESP_LOGW(TAG, "QBUF fail: %d", errno);
        }

        cur ^= 1;
    }
}
