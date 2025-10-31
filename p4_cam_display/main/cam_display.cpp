// cam_display.cpp — ESP32-P4 + OV5647 → GC9A01 (RGB565-only, FILL-only)
// Optimized: fixed-point scaling map + CPU/DMA overlap + FPS logger.

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

// ===== LCD & pins =====
static constexpr int LCD_W = 240;
static constexpr int LCD_H = 240;

static constexpr spi_host_device_t LCD_HOST = SPI2_HOST;
static constexpr int PIN_LCD_SCLK = 32;
static constexpr int PIN_LCD_MOSI = 26;
static constexpr int PIN_LCD_MISO = -1;
static constexpr int PIN_LCD_CS   = 25;
static constexpr int PIN_LCD_DC   = 27;
static constexpr int PIN_LCD_RST  = 20;
static constexpr int PIN_LCD_BL   = 21;

// 80 MHz is prima; als nodig 60/40 testen
static constexpr uint32_t LCD_SPI_HZ = (80u * 1000u * 1000u);

// FILL-crop finetune
static constexpr int   CALIB_SHIFT_X = 300;
static constexpr int   CALIB_SHIFT_Y = 30;
static constexpr float EXTRA_ZOOM    = 0.04f;

// ---- helpers ----
static inline gpio_num_t to_gpio(int pin){ return (pin >= 0) ? (gpio_num_t)pin : GPIO_NUM_NC; }
static inline uint16_t bswap16(uint16_t v){ return (uint16_t)((v<<8)|(v>>8)); }
static inline int clampi(int v,int lo,int hi){ return v<lo?lo:(v>hi?hi:v); }

// LCD DMA done semaphore
static SemaphoreHandle_t g_lcd_done = nullptr;
static bool on_color_trans_done_cb(esp_lcd_panel_io_handle_t, esp_lcd_panel_io_event_data_t*, void*)
{
    BaseType_t hpw = pdFALSE;
    if (g_lcd_done) xSemaphoreGiveFromISR(g_lcd_done, &hpw);
    return hpw == pdTRUE;
}

// Open cam
static int open_first_available_cam(char out_path[32])
{
    const char* try_list[] = { "/dev/mipi_csi_cam0","/dev/dvp_cam0","/dev/spi_cam0","/dev/uvc0","/dev/video0" };
    for (size_t i=0;i<sizeof(try_list)/sizeof(try_list[0]);++i){
        int fd = open(try_list[i], O_RDWR);
        if (fd>=0){ strncpy(out_path, try_list[i], 31); out_path[31]='\0'; return fd; }
    }
    out_path[0]='\0'; return -1;
}

// Build fixed-point map: dest[0..dstN-1] → src_start + floor((i*(srcN-1))/(dstN-1))
static void build_map_q16(int* map, int dstN, int src_start, int srcN)
{
    if (dstN<=1){ map[0]=src_start; return; }
    const uint32_t inc = ((uint32_t)(srcN-1) << 16) / (uint32_t)(dstN-1);
    uint32_t acc = 0;
    for (int i=0;i<dstN;++i){ map[i] = src_start + (int)(acc >> 16); acc += inc; }
}

extern "C" void app_main(void)
{
    // 1) Init video
    ESP_ERROR_CHECK(example_video_init());
    ESP_LOGI(TAG, "ESP-Video initialized");

    // 2) Open cam
    char used_dev[32];
    int fd = open_first_available_cam(used_dev);
    if (fd < 0){ ESP_LOGE(TAG,"Geen /dev/videoX device gevonden."); vTaskDelay(portMAX_DELAY); }
    ESP_LOGI(TAG, "Video device: %s", used_dev);

    // 3) Force RGB565 ~800x640
    v4l2_format set_fmt{};
    set_fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    set_fmt.fmt.pix.width        = 800;
    set_fmt.fmt.pix.height       = 640;
    set_fmt.fmt.pix.pixelformat  = V4L2_PIX_FMT_RGB565;
    set_fmt.fmt.pix.field        = V4L2_FIELD_NONE;
    (void)ioctl(fd, VIDIOC_S_FMT, &set_fmt);

    v4l2_format fmt{};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd, VIDIOC_G_FMT, &fmt) != 0){ ESP_LOGE(TAG,"VIDIOC_G_FMT fail: %d", errno); vTaskDelay(portMAX_DELAY); }

    if (fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_RGB565){
        ESP_LOGE(TAG,"Minimal build verwacht RGB565. FourCC=0x%08X", fmt.fmt.pix.pixelformat);
        vTaskDelay(portMAX_DELAY);
    }
    const int src_w = (int)fmt.fmt.pix.width;
    const int src_h = (int)fmt.fmt.pix.height;
    int stride_bytes = (int)fmt.fmt.pix.bytesperline;
    if (stride_bytes<=0) stride_bytes = src_w*2;

    ESP_LOGI(TAG, "Camera fmt: %dx%d RGB565, stride=%d", src_w, src_h, stride_bytes);

    // 4) Buffers & stream
    v4l2_requestbuffers req{}; req.count=4; req.type=V4L2_BUF_TYPE_VIDEO_CAPTURE; req.memory=V4L2_MEMORY_MMAP;
    if (ioctl(fd, VIDIOC_REQBUFS, &req)!=0 || req.count<2){ ESP_LOGE(TAG,"REQBUFS fail (count=%u)", req.count); vTaskDelay(portMAX_DELAY); }

    struct MMapBuf{ void* start; size_t length; } buffers[8]{};
    for (uint32_t i=0;i<req.count;++i){
        v4l2_buffer b{}; b.type=V4L2_BUF_TYPE_VIDEO_CAPTURE; b.memory=V4L2_MEMORY_MMAP; b.index=i;
        if (ioctl(fd, VIDIOC_QUERYBUF, &b)!=0){ ESP_LOGE(TAG,"QUERYBUF %u fail: %d", i, errno); vTaskDelay(portMAX_DELAY); }
        buffers[i].length=b.length;
        buffers[i].start =mmap(NULL,b.length,PROT_READ|PROT_WRITE,MAP_SHARED,fd,b.m.offset);
        if (buffers[i].start==MAP_FAILED){ ESP_LOGE(TAG,"mmap %u fail", i); vTaskDelay(portMAX_DELAY); }
        if (ioctl(fd, VIDIOC_QBUF, &b)!=0){ ESP_LOGE(TAG,"QBUF %u fail: %d", i, errno); vTaskDelay(portMAX_DELAY); }
    }
    int type=V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd, VIDIOC_STREAMON, &type)!=0){ ESP_LOGE(TAG,"STREAMON fail: %d", errno); vTaskDelay(portMAX_DELAY); }

    // 5) LCD init
    spi_bus_config_t buscfg{}; buscfg.sclk_io_num=PIN_LCD_SCLK; buscfg.mosi_io_num=PIN_LCD_MOSI; buscfg.miso_io_num=PIN_LCD_MISO;
    buscfg.quadwp_io_num=-1; buscfg.quadhd_io_num=-1; buscfg.max_transfer_sz=LCD_W*LCD_H*2+16;
    ESP_ERROR_CHECK(spi_bus_initialize(LCD_HOST, &buscfg, SPI_DMA_CH_AUTO));

    if (PIN_LCD_BL>=0){ gpio_config_t io{}; io.pin_bit_mask=(1ULL<<(unsigned)PIN_LCD_BL); io.mode=GPIO_MODE_OUTPUT; gpio_config(&io); gpio_set_level(to_gpio(PIN_LCD_BL),1); }
    if (PIN_LCD_SCLK>=0) gpio_set_drive_capability(to_gpio(PIN_LCD_SCLK), GPIO_DRIVE_CAP_3);
    if (PIN_LCD_MOSI>=0) gpio_set_drive_capability(to_gpio(PIN_LCD_MOSI), GPIO_DRIVE_CAP_3);
    if (PIN_LCD_DC  >=0) gpio_set_drive_capability(to_gpio(PIN_LCD_DC),   GPIO_DRIVE_CAP_3);
    if (PIN_LCD_CS  >=0) gpio_set_drive_capability(to_gpio(PIN_LCD_CS),   GPIO_DRIVE_CAP_3);

    g_lcd_done = xSemaphoreCreateBinary(); configASSERT(g_lcd_done);

    esp_lcd_panel_io_handle_t io_handle=nullptr;
    esp_lcd_panel_io_spi_config_t io_cfg{}; io_cfg.dc_gpio_num=to_gpio(PIN_LCD_DC); io_cfg.cs_gpio_num=to_gpio(PIN_LCD_CS);
    io_cfg.pclk_hz=LCD_SPI_HZ; io_cfg.lcd_cmd_bits=8; io_cfg.lcd_param_bits=8; io_cfg.spi_mode=0;
    io_cfg.trans_queue_depth=2;                      // ← allow overlap (2 in-flight)
    io_cfg.on_color_trans_done=on_color_trans_done_cb;
    ESP_ERROR_CHECK(esp_lcd_new_panel_io_spi((esp_lcd_spi_bus_handle_t)LCD_HOST, &io_cfg, &io_handle));

    esp_lcd_panel_handle_t panel=nullptr;
    esp_lcd_panel_dev_config_t panel_cfg{}; panel_cfg.reset_gpio_num=to_gpio(PIN_LCD_RST);
    panel_cfg.rgb_ele_order=LCD_RGB_ELEMENT_ORDER_BGR; panel_cfg.bits_per_pixel=16;
    ESP_ERROR_CHECK(esp_lcd_new_panel_gc9a01(io_handle, &panel_cfg, &panel));

    ESP_ERROR_CHECK(esp_lcd_panel_reset(panel)); vTaskDelay(pdMS_TO_TICKS(20));
    ESP_ERROR_CHECK(esp_lcd_panel_init(panel));  vTaskDelay(pdMS_TO_TICKS(120));
    ESP_ERROR_CHECK(esp_lcd_panel_disp_on_off(panel,true));
    ESP_ERROR_CHECK(esp_lcd_panel_invert_color(panel,true));
    ESP_ERROR_CHECK(esp_lcd_panel_swap_xy(panel,false));
    ESP_ERROR_CHECK(esp_lcd_panel_mirror(panel,false,false));
    ESP_ERROR_CHECK(esp_lcd_panel_set_gap(panel,0,0));

    // 6) Framebuffers
    const size_t FB_PIXELS=(size_t)LCD_W*LCD_H, FB_BYTES=FB_PIXELS*2;
    uint16_t* fb[2]={nullptr,nullptr};
    for(int i=0;i<2;++i){ fb[i]=(uint16_t*)heap_caps_malloc(FB_BYTES, MALLOC_CAP_DMA|MALLOC_CAP_INTERNAL); configASSERT(fb[i]); }
    int cur=0;

    // 7) ROI + precomputed maps (fixed-point) — build once
    const int base_sq = (src_w<src_h)?src_w:src_h;
    int extra = (int)(base_sq*EXTRA_ZOOM + 0.5f);
    int roi_size = base_sq - extra; if (roi_size<16) roi_size=16;

    int x0 = (src_w - roi_size)/2;
    int y0 = (src_h - roi_size)/2;
    x0 = clampi(x0 + CALIB_SHIFT_X, 0, src_w - roi_size);
    y0 = clampi(y0 + CALIB_SHIFT_Y, 0, src_h - roi_size);

    const int roi_w=roi_size, roi_h=roi_size;
    ESP_LOGI(TAG, "FILL ROI=%d @(%d,%d) from %dx%d", roi_size, x0, y0, src_w, src_h);

    static int xmap[LCD_W], ymap[LCD_H];
    build_map_q16(xmap, LCD_W, x0, roi_w);
    build_map_q16(ymap, LCD_H, y0, roi_h);

    // ===== FPS & timings =====
    uint64_t t_fps0 = esp_timer_get_time();
    int      fps_frames=0;
    double   avg_scale_ms=0.0, avg_xfer_ms=0.0;
    const double alpha=0.2;

    // DMA in-flight tracking: wait right before scheduling the next draw
    bool dma_active=false;

    // 8) Loop
    while(true){
        // Dequeue a frame
        v4l2_buffer buf{}; buf.type=V4L2_BUF_TYPE_VIDEO_CAPTURE; buf.memory=V4L2_MEMORY_MMAP;
        if (ioctl(fd, VIDIOC_DQBUF, &buf)!=0){ ESP_LOGW(TAG,"DQBUF fail: %d", errno); vTaskDelay(pdMS_TO_TICKS(1)); continue; }
        if (!(buf.flags & V4L2_BUF_FLAG_DONE)){ (void)ioctl(fd, VIDIOC_QBUF, &buf); continue; }

        const uint8_t* frame = (const uint8_t*)buffers[buf.index].start;
        uint16_t* dst = fb[cur];

        uint64_t t0 = esp_timer_get_time();

        // Scale using prebuilt maps (no divides in inner loop)
        for (int dy=0; dy<LCD_H; ++dy){
            const int sy = ymap[dy];
            const uint16_t* src_line = (const uint16_t*)(frame + (size_t)sy * (size_t)stride_bytes);
            uint16_t* out_line = dst + (size_t)dy * LCD_W;

            // Unroll a bit for throughput
            for (int dx=0; dx<LCD_W; ++dx){
                const int sx = xmap[dx];
                out_line[dx] = bswap16(src_line[sx]);
            }
        }

        uint64_t t1 = esp_timer_get_time();

        // If previous DMA still active, wait now (overlap achieved)
        if (dma_active){
            xSemaphoreTake(g_lcd_done, pdMS_TO_TICKS(50));
            dma_active=false;
        }

        // Kick DMA for this buffer (non-blocking)
        ESP_ERROR_CHECK(esp_lcd_panel_draw_bitmap(panel, 0, 0, LCD_W, LCD_H, dst));
        dma_active=true;

        uint64_t t2 = esp_timer_get_time();

        // Requeue capture buffer ASAP (keeps camera running)
        if (ioctl(fd, VIDIOC_QBUF, &buf)!=0){ ESP_LOGW(TAG,"QBUF fail: %d", errno); }

        // timings
        double scale_ms = (t1 - t0)/1000.0;
        double xfer_ms  = (t2 - t1)/1000.0; // nominal (submit cost); real DMA hidden by overlap
        avg_scale_ms = (avg_scale_ms==0.0)? scale_ms : (alpha*scale_ms + (1.0-alpha)*avg_scale_ms);
        avg_xfer_ms  = (avg_xfer_ms==0.0)? xfer_ms  : (alpha*xfer_ms  + (1.0-alpha)*avg_xfer_ms);

        // FPS (based on full loop cadence)
        fps_frames++;
        uint64_t now = t2;
        if (now - t_fps0 >= 1000000ULL){
            double secs = (now - t_fps0)/1000000.0;
            double fps  = fps_frames/secs;
            ESP_LOGI(TAG, "FPS: %.1f | scale: %.2f ms | submit: %.2f ms (DMA overlapped)",
                     fps, avg_scale_ms, avg_xfer_ms);
            fps_frames=0; t_fps0=now;
        }

        // swap fb
        cur ^= 1;
    }
}
