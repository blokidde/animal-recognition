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

#include "driver/spi_master.h"
#include "driver/gpio.h"

#include "esp_lcd_panel_ops.h"
#include "esp_lcd_panel_io.h"
#include "esp_lcd_io_spi.h"
#include "esp_lcd_gc9a01.h"

// -------- ESP-Video / V4L2 ----------
// Provides V4L2-based capture helpers for ESP-Video example devices
#include "example_video_common.h"
#include <linux/videodev2.h>

static const char* TAG = "P4_CAM_GC9A01_MIN_FAST_SAFE";

// ===== LCD & pins =====
// Physical LCD dimensions (GC9A01 round panel is 240x240)
static constexpr int LCD_W = 240;
static constexpr int LCD_H = 240;

// SPI host and pin mapping for the LCD panel
static constexpr spi_host_device_t LCD_HOST = SPI2_HOST;
static constexpr int PIN_LCD_SCLK = 32;
static constexpr int PIN_LCD_MOSI = 26;
static constexpr int PIN_LCD_MISO = -1; // write-only
static constexpr int PIN_LCD_CS   = 25;
static constexpr int PIN_LCD_DC   = 27;
static constexpr int PIN_LCD_RST  = 20;
static constexpr int PIN_LCD_BL   = 21;

// SPI clock: 80 MHz is fast and generally stable. 100 MHz produced errors → avoid it.
static constexpr uint32_t LCD_SPI_HZ = (80u * 1000u * 1000u);

// FILL-crop fine-tuning (software centering and slight zoom)
static constexpr int   CALIB_SHIFT_X = 300;
static constexpr int   CALIB_SHIFT_Y = 30;
static constexpr float EXTRA_ZOOM    = 0.04f;

// ---- GC9A01 command set ----
#define GC9A01_CMD_CASET 0x2A
#define GC9A01_CMD_RASET 0x2B
#define GC9A01_CMD_RAMWR 0x2C

// ---- small helpers ----
static inline gpio_num_t to_gpio(int pin){ return (pin >= 0) ? (gpio_num_t)pin : GPIO_NUM_NC; }
static inline uint16_t   bswap16(uint16_t v){ return (uint16_t)((v << 8) | (v >> 8)); }
static inline int        clampi(int v,int lo,int hi){ return v<lo?lo:(v>hi?hi:v); }

// ---------- LCD transfer completion sync ----------
// We serialize full-frame transfers by keeping the queue depth at 1 and waiting
// for a done-semaphore raised by this ISR callback. This guarantees “tear-safe” swaps.
static SemaphoreHandle_t g_lcd_done = nullptr;
static bool on_color_trans_done_cb(esp_lcd_panel_io_handle_t,
                                   esp_lcd_panel_io_event_data_t*,
                                   void*)
{
    BaseType_t hpw = pdFALSE;
    if (g_lcd_done) xSemaphoreGiveFromISR(g_lcd_done, &hpw);
    return hpw == pdTRUE;
}

// ---------- Camera open helper ----------
// Try a list of known video device nodes and return the first that opens.
static int open_first_available_cam(char out_path[32])
{
    const char* try_list[] = {
        "/dev/mipi_csi_cam0","/dev/dvp_cam0","/dev/spi_cam0","/dev/uvc0","/dev/video0"
    };
    for (size_t i=0;i<sizeof(try_list)/sizeof(try_list[0]);++i){
        int fd = open(try_list[i], O_RDWR);
        if (fd>=0){ strncpy(out_path, try_list[i], 31); out_path[31]='\0'; return fd; }
    }
    out_path[0]='\0'; return -1;
}

// Build mapping: dst i in [0..outN-1] → src_start + floor((i*(srcN-1))/(outN-1))
// (guarantees i=0→src_start and i=outN-1→src_start+srcN-1; no edge off-by-ones)
static void build_map_q16(int* map, int outN, int src_start, int srcN)
{
    if (outN <= 1){ map[0] = src_start; return; }
    const uint32_t inc = ((uint32_t)(srcN - 1) << 16) / (uint32_t)(outN - 1);
    uint32_t acc = 0;
    for (int i=0; i<outN; ++i){ map[i] = src_start + (int)(acc >> 16); acc += inc; }
}

extern "C" void app_main(void)
{
    // 1) Initialize ESP-Video framework (V4L2 plumbing, etc.)
    ESP_ERROR_CHECK(example_video_init());
    ESP_LOGI(TAG, "ESP-Video initialized");

    // 2) Open the first available camera device
    char used_dev[32];
    int fd = open_first_available_cam(used_dev);
    if (fd < 0){ ESP_LOGE(TAG,"Geen /dev/videoX device gevonden."); vTaskDelay(portMAX_DELAY); }
    ESP_LOGI(TAG, "Video device: %s", used_dev);

    // 3) Force RGB565 at ~800x640 (fast for scaling and matches LCD color depth)
    struct v4l2_format set_fmt{}; set_fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    set_fmt.fmt.pix.width        = 800;
    set_fmt.fmt.pix.height       = 640;
    set_fmt.fmt.pix.pixelformat  = V4L2_PIX_FMT_RGB565;
    set_fmt.fmt.pix.field        = V4L2_FIELD_NONE;
    (void)ioctl(fd, VIDIOC_S_FMT, &set_fmt);

    struct v4l2_format fmt{}; fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd, VIDIOC_G_FMT, &fmt) != 0){ ESP_LOGE(TAG,"VIDIOC_G_FMT fail: %d", errno); vTaskDelay(portMAX_DELAY); }

    if (fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_RGB565){
        ESP_LOGE(TAG,"RGB565 verwacht. FourCC=0x%08X", fmt.fmt.pix.pixelformat);
        vTaskDelay(portMAX_DELAY);
    }
    const int src_w = (int)fmt.fmt.pix.width;
    const int src_h = (int)fmt.fmt.pix.height;
    int stride_bytes = (int)fmt.fmt.pix.bytesperline;
    if (stride_bytes<=0) stride_bytes = src_w*2;

    ESP_LOGI(TAG, "Camera fmt: %dx%d RGB565, stride=%d", src_w, src_h, stride_bytes);

    // 4) Request and mmap capture buffers, then start streaming
    struct v4l2_requestbuffers req{}; req.count=4; req.type=V4L2_BUF_TYPE_VIDEO_CAPTURE; req.memory=V4L2_MEMORY_MMAP;
    if (ioctl(fd, VIDIOC_REQBUFS, &req)!=0 || req.count<2){ ESP_LOGE(TAG,"REQBUFS fail (count=%u)", req.count); vTaskDelay(portMAX_DELAY); }

    struct MMapBuf{ void* start; size_t length; } buffers[8]{};
    for (uint32_t i=0;i<req.count;++i){
        struct v4l2_buffer b{}; b.type=V4L2_BUF_TYPE_VIDEO_CAPTURE; b.memory=V4L2_MEMORY_MMAP; b.index=i;
        if (ioctl(fd, VIDIOC_QUERYBUF, &b)!=0){ ESP_LOGE(TAG,"QUERYBUF %u fail: %d", i, errno); vTaskDelay(portMAX_DELAY); }
        buffers[i].length=b.length;
        buffers[i].start =mmap(NULL,b.length,PROT_READ|PROT_WRITE,MAP_SHARED,fd,b.m.offset);
        if (buffers[i].start==MAP_FAILED){ ESP_LOGE(TAG,"mmap %u fail", i); vTaskDelay(portMAX_DELAY); }
        if (ioctl(fd, VIDIOC_QBUF, &b)!=0){ ESP_LOGE(TAG,"QBUF %u fail: %d", i, errno); vTaskDelay(portMAX_DELAY); }
    }
    int type=V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd, VIDIOC_STREAMON, &type)!=0){ ESP_LOGE(TAG,"STREAMON fail: %d", errno); vTaskDelay(portMAX_DELAY); }

    // 5) LCD init (single transfer in-flight) + custom IO handle (we send CASET/RASET/RAMWR ourselves)
    spi_bus_config_t buscfg{};
    buscfg.sclk_io_num     = PIN_LCD_SCLK;
    buscfg.mosi_io_num     = PIN_LCD_MOSI;
    buscfg.miso_io_num     = PIN_LCD_MISO;
    buscfg.quadwp_io_num   = -1;
    buscfg.quadhd_io_num   = -1;
    // ensure ONE FULL frame fits in a single DMA transfer (reduces tearing risks)
    buscfg.max_transfer_sz = LCD_W * LCD_H * (int)sizeof(uint16_t) + 64; // small margin
    ESP_ERROR_CHECK(spi_bus_initialize(LCD_HOST, &buscfg, SPI_DMA_CH_AUTO));

    // Backlight GPIO, enable if present
    if (PIN_LCD_BL >= 0){
        gpio_config_t io{}; io.pin_bit_mask = (1ULL << (unsigned)PIN_LCD_BL); io.mode = GPIO_MODE_OUTPUT;
        gpio_config(&io); gpio_set_level(to_gpio(PIN_LCD_BL), 1);
    }
    // Increase drive strength on high-speed SPI signals
    if (PIN_LCD_SCLK>=0) gpio_set_drive_capability(to_gpio(PIN_LCD_SCLK), GPIO_DRIVE_CAP_3);
    if (PIN_LCD_MOSI>=0) gpio_set_drive_capability(to_gpio(PIN_LCD_MOSI), GPIO_DRIVE_CAP_3);
    if (PIN_LCD_DC  >=0) gpio_set_drive_capability(to_gpio(PIN_LCD_DC),   GPIO_DRIVE_CAP_3);
    if (PIN_LCD_CS  >=0) gpio_set_drive_capability(to_gpio(PIN_LCD_CS),   GPIO_DRIVE_CAP_3);

    // Create semaphore to wait for DMA completion (ISR gives it)
    g_lcd_done = xSemaphoreCreateBinary(); configASSERT(g_lcd_done);

    // Create SPI panel IO (queue depth = 1 for strict sequencing)
    esp_lcd_panel_io_handle_t io_handle=nullptr;
    esp_lcd_panel_io_spi_config_t io_cfg{};
    io_cfg.dc_gpio_num         = to_gpio(PIN_LCD_DC);
    io_cfg.cs_gpio_num         = to_gpio(PIN_LCD_CS);
    io_cfg.pclk_hz             = LCD_SPI_HZ;
    io_cfg.lcd_cmd_bits        = 8;
    io_cfg.lcd_param_bits      = 8;
    io_cfg.spi_mode            = 0;               // alternative: mode 3
    io_cfg.trans_queue_depth   = 1;               // <<< core: exactly 1 in-flight transfer
    io_cfg.on_color_trans_done = on_color_trans_done_cb;
    ESP_ERROR_CHECK(esp_lcd_new_panel_io_spi((esp_lcd_spi_bus_handle_t)LCD_HOST, &io_cfg, &io_handle));

    // We only use a "panel" handle for initialization; pixel writes are manual via io_handle
    esp_lcd_panel_handle_t panel=nullptr;
    esp_lcd_panel_dev_config_t panel_cfg{};
    panel_cfg.reset_gpio_num = to_gpio(PIN_LCD_RST);
    panel_cfg.rgb_ele_order  = LCD_RGB_ELEMENT_ORDER_BGR;   // GC9A01 often expects BGR
    panel_cfg.bits_per_pixel = 16;
    ESP_ERROR_CHECK(esp_lcd_new_panel_gc9a01(io_handle, &panel_cfg, &panel));

    ESP_ERROR_CHECK(esp_lcd_panel_reset(panel)); vTaskDelay(pdMS_TO_TICKS(20));
    ESP_ERROR_CHECK(esp_lcd_panel_init(panel));  vTaskDelay(pdMS_TO_TICKS(120));
    ESP_ERROR_CHECK(esp_lcd_panel_disp_on_off(panel,true));
    ESP_ERROR_CHECK(esp_lcd_panel_invert_color(panel,true)); // if border artifacts: try false
    ESP_ERROR_CHECK(esp_lcd_panel_swap_xy(panel,false));
    ESP_ERROR_CHECK(esp_lcd_panel_mirror(panel,false,false));
    ESP_ERROR_CHECK(esp_lcd_panel_set_gap(panel,0,0));

    // Helper: set full address window every frame (resets write pointer)
    auto set_full_window = [&](void){
        uint8_t caset[4] = { 0x00, 0x00, 0x00, (uint8_t)(LCD_W-1) };
        uint8_t raset[4] = { 0x00, 0x00, 0x00, (uint8_t)(LCD_H-1) };
        ESP_ERROR_CHECK(esp_lcd_panel_io_tx_param(io_handle, GC9A01_CMD_CASET, caset, sizeof(caset)));
        ESP_ERROR_CHECK(esp_lcd_panel_io_tx_param(io_handle, GC9A01_CMD_RASET, raset, sizeof(raset)));
    };

    // 6) Allocate two DMA-capable framebuffers in internal RAM (ping-pong)
    const size_t FB_PIXELS = (size_t)LCD_W * LCD_H;
    const size_t FB_BYTES  = FB_PIXELS * sizeof(uint16_t);
    uint16_t* fb[2] = {nullptr, nullptr};
    for (int i=0; i<2; ++i) {
        fb[i] = (uint16_t*) heap_caps_malloc(FB_BYTES, MALLOC_CAP_DMA | MALLOC_CAP_INTERNAL);
        configASSERT(fb[i] && "kon geen INTERNAL DMA buffer allocaten");
    }
    int cur = 0;

    // 7) Determine square FILL ROI and precompute X/Y mapping tables
    int base_sq  = (src_w < src_h) ? src_w : src_h;
    int extra    = (int)((float)base_sq * EXTRA_ZOOM + 0.5f);
    int roi_size = base_sq - extra; if (roi_size < 16) roi_size = 16;

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

    // ===== FPS & timings =====
    // Exponential moving averages for scale (CPU time) and submit (command/queue time)
    uint64_t t_fps0 = esp_timer_get_time();
    int      fps_frames=0;
    double   avg_scale_ms=0.0, avg_submit_ms=0.0;
    const double alpha=0.2;

    bool dma_active = false;

    // 8) Main loop: capture → scale/copy → wait previous DMA → submit new frame → stats
    while (true) {
        // Dequeue a captured frame
        struct v4l2_buffer buf{}; buf.type=V4L2_BUF_TYPE_VIDEO_CAPTURE; buf.memory=V4L2_MEMORY_MMAP;
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

        // === Scale: ROI (RGB565) -> 240x240 via precomputed maps (fast nearest) ===
        uint64_t t0 = esp_timer_get_time();
        for (int dy=0; dy<LCD_H; ++dy) {
            const int sy = ymap[dy];
            const uint16_t* src_line = (const uint16_t*)(frame + (size_t)sy * (size_t)stride_bytes);
            uint16_t* out_line = dst + (size_t)dy * LCD_W;

            int dx = 0;
            // Unrolled 8-pixel chunk for throughput; bswap16 matches panel endianness
            for (; dx <= LCD_W - 8; dx += 8) {
                out_line[dx+0] = bswap16(src_line[xmap[dx+0]]);
                out_line[dx+1] = bswap16(src_line[xmap[dx+1]]);
                out_line[dx+2] = bswap16(src_line[xmap[dx+2]]);
                out_line[dx+3] = bswap16(src_line[xmap[dx+3]]);
                out_line[dx+4] = bswap16(src_line[xmap[dx+4]]);
                out_line[dx+5] = bswap16(src_line[xmap[dx+5]]);
                out_line[dx+6] = bswap16(src_line[xmap[dx+6]]);
                out_line[dx+7] = bswap16(src_line[xmap[dx+7]]);
            }
            // Tail elements
            for (; dx < LCD_W; ++dx) {
                out_line[dx] = bswap16(src_line[xmap[dx]]);
            }
        }
        uint64_t t1 = esp_timer_get_time();

        // === Precisely now: wait for previous frame to finish before queuing the next ===
        if (dma_active) {
            xSemaphoreTake(g_lcd_done, pdMS_TO_TICKS(50));
            dma_active = false;
        }

        // Reset address window each frame (avoids write-pointer drift), then push one full-color transfer
        set_full_window();
        ESP_ERROR_CHECK(esp_lcd_panel_io_tx_color(io_handle, GC9A01_CMD_RAMWR, dst, FB_BYTES));
        dma_active = true; // callback fires when the whole frame is sent

        uint64_t t2 = esp_timer_get_time();

        // Re-queue the capture buffer back to the driver
        if (ioctl(fd, VIDIOC_QBUF, &buf) != 0) { ESP_LOGW(TAG, "QBUF fail: %d", errno); }

        // Timings: submit is command/queue cost; wire-time overlaps with next scaling step
        double scale_ms  = (t1 - t0)/1000.0;
        double submit_ms = (t2 - t1)/1000.0;
        avg_scale_ms  = (avg_scale_ms==0.0)? scale_ms  : (alpha*scale_ms  + (1.0-alpha)*avg_scale_ms);
        avg_submit_ms = (avg_submit_ms==0.0)? submit_ms : (alpha*submit_ms + (1.0-alpha)*avg_submit_ms);

        // FPS logging once per ~second
        fps_frames++;
        uint64_t now = t2;
        if (now - t_fps0 >= 1000000ULL){
            double secs = (now - t_fps0)/1000000.0;
            double fps  = fps_frames/secs;
            ESP_LOGI(TAG, "FPS: %.1f | scale: %.2f ms | submit: %.2f ms (QD=1, full-frame color xfer)",
                     fps, avg_scale_ms, avg_submit_ms);
            fps_frames=0; t_fps0=now;
        }

        // Swap framebuffer for next iteration
        cur ^= 1;
    }
}
