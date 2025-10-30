// cam_display.cpp  — ESP32-P4 + OV5647 (ESP-Video) -> GC9A01 240x240 SPI LCD
#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/param.h>
#include <unistd.h>
#include <stdint.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "esp_assert.h"

#include "esp_system.h"
#include "esp_mac.h"                // <-- IDF 5.5 hint: niet meer ge-include via esp_system.h

#include "driver/spi_master.h"
#include "driver/gpio.h"

#include "esp_lcd_panel_ops.h"
#include "esp_lcd_panel_io.h"
#include "esp_lcd_io_spi.h"
#include "esp_lcd_gc9a01.h"

// -------- ESP-Video / V4L2 ----------
#include "example_video_common.h"
#include <linux/videodev2.h>

static const char* TAG = "P4_CAM_GC9A01";

#ifndef CAM_PWR_EN_GPIO
#define CAM_PWR_EN_GPIO  (-1)
#endif

// ---- LCD pins/params ----
static constexpr spi_host_device_t LCD_HOST = SPI2_HOST;
static constexpr int PIN_LCD_SCLK = 32;
static constexpr int PIN_LCD_MOSI = 26;
static constexpr int PIN_LCD_MISO = -1; // write-only
static constexpr int PIN_LCD_CS   = 25;
static constexpr int PIN_LCD_DC   = 27;
static constexpr int PIN_LCD_RST  = 20; // -1 als geen RST
static constexpr int PIN_LCD_BL   = 21; // -1 als geen backlight

static constexpr int LCD_W = 240;
static constexpr int LCD_H = 240;
static constexpr uint32_t LCD_SPI_HZ = (80u * 1000u * 1000u); // 80 MHz; verlaag naar 60 MHz als je artefacts ziet

// Oriëntatie
static constexpr bool ORIENT_SWAP_XY  = false;
static constexpr bool ORIENT_MIRROR_X = true;
static constexpr bool ORIENT_MIRROR_Y = false;

static inline gpio_num_t to_gpio(int pin) { return (pin >= 0) ? static_cast<gpio_num_t>(pin) : GPIO_NUM_NC; }
static inline uint16_t bswap16(uint16_t v) { return (uint16_t)((v << 8) | (v >> 8)); }

// ---- Device-pad kiezen op basis van config ----
#if CONFIG_EXAMPLE_ENABLE_MIPI_CSI_CAM_SENSOR
static constexpr const char* kDevMIPI = "/dev/mipi_csi_cam0";
#endif
#if CONFIG_EXAMPLE_ENABLE_DVP_CAM_SENSOR
static constexpr const char* kDevDVP  = "/dev/dvp_cam0";
#endif
#if CONFIG_EXAMPLE_ENABLE_SPI_CAM_SENSOR
static constexpr const char* kDevSPI  = "/dev/spi_cam0";
#endif
#if CONFIG_EXAMPLE_ENABLE_USB_UVC_CAM_SENSOR
static constexpr const char* kDevUVC0 = "/dev/uvc0";
#endif
static constexpr const char* kDevFallback = "/dev/video0";

static int open_first_available_cam(char out_path[32])
{
    const char* try_list[8] = {0};
    int idx = 0;

    #if CONFIG_EXAMPLE_ENABLE_MIPI_CSI_CAM_SENSOR
    try_list[idx++] = kDevMIPI;
    #endif
    #if CONFIG_EXAMPLE_ENABLE_DVP_CAM_SENSOR
    try_list[idx++] = kDevDVP;
    #endif
    #if CONFIG_EXAMPLE_ENABLE_SPI_CAM_SENSOR
    try_list[idx++] = kDevSPI;
    #endif
    #if CONFIG_EXAMPLE_ENABLE_USB_UVC_CAM_SENSOR
    try_list[idx++] = kDevUVC0;
    #endif

    try_list[idx++] = "/dev/mipi_csi_cam0";
    try_list[idx++] = "/dev/dvp_cam0";
    try_list[idx++] = "/dev/spi_cam0";
    try_list[idx++] = kDevFallback;

    for (int i=0; i<idx; ++i) {
        if (!try_list[i]) continue;
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

// ---------- YUYV helpers ----------
static inline uint16_t pack_rgb565(int r, int g, int b) {
    return (uint16_t)(((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3));
}
static inline int clip8(int x){ return x < 0 ? 0 : (x > 255 ? 255 : x); }

static inline uint16_t yuyv_at_to_rgb565(const uint8_t* line, int src_w, int x_even_or_odd)
{
    int pair_x = x_even_or_odd & ~1;
    const uint8_t* p = line + pair_x*2; // [Y0 U Y1 V]
    int Y = p[(x_even_or_odd & 1) ? 2 : 0];
    int U = p[1];
    int V = p[3];
    int C = Y - 16;
    int D = U - 128;
    int E = V - 128;
    int r = clip8((298 * C + 409 * E + 128) >> 8);
    int g = clip8((298 * C - 100 * D - 208 * E + 128) >> 8);
    int b = clip8((298 * C + 516 * D + 128) >> 8);
    return pack_rgb565(r, g, b);
}

extern "C" void app_main(void)
{
    // 1) Init ESP-Video
    ESP_ERROR_CHECK(example_video_init());
    ESP_LOGI(TAG, "ESP-Video initialized");

    // 2) Open camera device
    char used_dev[32];
    int fd = open_first_available_cam(used_dev);
    if (fd < 0) {
        ESP_LOGE(TAG,
                 "Geen video device te openen. Controleer menuconfig en HW:\n"
                 " - Example Video Initialization -> MIPI-CSI ENABLED (OV5647)\n"
                 " - SCCB I2C pins correct (SCL=8, SDA=7)\n"
                 " - Camera op CSI-connector, kabel oriëntatie OK.");
        vTaskDelay(portMAX_DELAY);
    }
    ESP_LOGI(TAG, "Video device: %s", used_dev);

    // 3) Probeer eerst 240x240 RGB565 te krijgen
    v4l2_format set_fmt{};
    set_fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    set_fmt.fmt.pix.width        = 800;
    set_fmt.fmt.pix.height       = 800;
    set_fmt.fmt.pix.pixelformat  = V4L2_PIX_FMT_RGB565;
    set_fmt.fmt.pix.field        = V4L2_FIELD_NONE;
    (void)ioctl(fd, VIDIOC_S_FMT, &set_fmt); // als dit faalt, we regelen fallback hierna

    // Haal het uiteindelijke formaat op
    v4l2_format fmt{};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ESP_ERROR_CHECK(ioctl(fd, VIDIOC_G_FMT, &fmt));
    int src_w = (int)fmt.fmt.pix.width;
    int src_h = (int)fmt.fmt.pix.height;
    uint32_t fourcc = fmt.fmt.pix.pixelformat;

    bool cam_rgb565 = (fourcc == V4L2_PIX_FMT_RGB565);
    bool cam_yuyv   = (fourcc == V4L2_PIX_FMT_YUYV);

    ESP_LOGI(TAG, "Camera fmt: %dx%d fourcc=0x%08X (%s)",
             src_w, src_h, fourcc, cam_rgb565 ? "RGB565" : (cam_yuyv ? "YUYV" : "OTHER"));

    if (!(cam_rgb565 || cam_yuyv)) {
        // probeer YUYV (veelvoorkomend fallback)
        set_fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
        (void)ioctl(fd, VIDIOC_S_FMT, &set_fmt);
        ESP_ERROR_CHECK(ioctl(fd, VIDIOC_G_FMT, &fmt));
        src_w = (int)fmt.fmt.pix.width;
        src_h = (int)fmt.fmt.pix.height;
        fourcc = fmt.fmt.pix.pixelformat;
        cam_rgb565 = (fourcc == V4L2_PIX_FMT_RGB565);
        cam_yuyv   = (fourcc == V4L2_PIX_FMT_YUYV);
        ESP_LOGI(TAG, "Fallback fmt: %dx%d fourcc=0x%08X", src_w, src_h, fourcc);
        if (!(cam_rgb565 || cam_yuyv)) {
            ESP_LOGE(TAG, "Unsupported camera pixel format. Pas sensor/ISP config aan.");
            vTaskDelay(portMAX_DELAY);
        }
    }

    // 4) V4L2 buffers + STREAMON
    v4l2_requestbuffers req{};
    req.count  = 4;
    req.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    ESP_ERROR_CHECK(ioctl(fd, VIDIOC_REQBUFS, &req));
    if (req.count < 2) {
        ESP_LOGE(TAG, "Onvoldoende v4l2 buffers");
        vTaskDelay(portMAX_DELAY);
    }

    struct MMapBuf { void* start; size_t length; } buffers[8] = {};
    for (uint32_t i=0; i<req.count; ++i) {
        v4l2_buffer buf{};
        buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index  = i;
        ESP_ERROR_CHECK(ioctl(fd, VIDIOC_QUERYBUF, &buf));

        buffers[i].length = buf.length;
        buffers[i].start  = mmap(NULL, buf.length, PROT_READ|PROT_WRITE, MAP_SHARED, fd, buf.m.offset);
        if (buffers[i].start == MAP_FAILED) {
            ESP_LOGE(TAG, "mmap failed voor buffer %u", i);
            vTaskDelay(portMAX_DELAY);
        }
        ESP_ERROR_CHECK(ioctl(fd, VIDIOC_QBUF, &buf));
    }
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ESP_ERROR_CHECK(ioctl(fd, VIDIOC_STREAMON, &type));

    // 5) Init LCD (GC9A01) – queue depth = 1 (geen overlap/tearing)
    spi_bus_config_t buscfg{};
    buscfg.sclk_io_num     = PIN_LCD_SCLK;
    buscfg.mosi_io_num     = PIN_LCD_MOSI;
    buscfg.miso_io_num     = PIN_LCD_MISO;
    buscfg.quadwp_io_num   = -1;
    buscfg.quadhd_io_num   = -1;
    buscfg.max_transfer_sz = LCD_W * LCD_H * (int)sizeof(uint16_t);
    ESP_ERROR_CHECK(spi_bus_initialize(LCD_HOST, &buscfg, SPI_DMA_CH_AUTO));

    if (PIN_LCD_BL >= 0) {
        gpio_config_t io_conf{};
        io_conf.pin_bit_mask = (1ULL << (unsigned)PIN_LCD_BL);
        io_conf.mode = GPIO_MODE_OUTPUT;
        gpio_config(&io_conf);
        gpio_set_level(to_gpio(PIN_LCD_BL), 1);
    }

    esp_lcd_panel_io_handle_t io_handle = nullptr;
    esp_lcd_panel_io_spi_config_t io_cfg{};
    io_cfg.dc_gpio_num       = to_gpio(PIN_LCD_DC);
    io_cfg.cs_gpio_num       = to_gpio(PIN_LCD_CS);
    io_cfg.pclk_hz           = LCD_SPI_HZ;
    io_cfg.lcd_cmd_bits      = 8;
    io_cfg.lcd_param_bits    = 8;
    io_cfg.spi_mode          = 0;
    io_cfg.trans_queue_depth = 1;  // <<< exact 1 in-flight transfer
    ESP_ERROR_CHECK(esp_lcd_new_panel_io_spi((esp_lcd_spi_bus_handle_t)LCD_HOST, &io_cfg, &io_handle));

    esp_lcd_panel_handle_t panel = nullptr;
    esp_lcd_panel_dev_config_t panel_cfg{};
    panel_cfg.reset_gpio_num = to_gpio(PIN_LCD_RST);
    panel_cfg.rgb_ele_order  = LCD_RGB_ELEMENT_ORDER_BGR;   // GC9A01 modules vaak BGR
    panel_cfg.bits_per_pixel = 16;                          // RGB565
    ESP_ERROR_CHECK(esp_lcd_new_panel_gc9a01(io_handle, &panel_cfg, &panel)); // <-- FIX: juiste signature (3 args)

    ESP_ERROR_CHECK(esp_lcd_panel_reset(panel));
    vTaskDelay(pdMS_TO_TICKS(20));
    ESP_ERROR_CHECK(esp_lcd_panel_init(panel));
    vTaskDelay(pdMS_TO_TICKS(120));
    ESP_ERROR_CHECK(esp_lcd_panel_disp_on_off(panel, true));
    ESP_ERROR_CHECK(esp_lcd_panel_invert_color(panel, true));
    ESP_ERROR_CHECK(esp_lcd_panel_swap_xy(panel, ORIENT_SWAP_XY));
    ESP_ERROR_CHECK(esp_lcd_panel_mirror(panel, ORIENT_MIRROR_X, ORIENT_MIRROR_Y));
    ESP_ERROR_CHECK(esp_lcd_panel_set_gap(panel, 0, 0));

    // 6) Double framebuffer (DMA-capable)
    const size_t FB_PIXELS = (size_t)LCD_W * LCD_H;
    const size_t FB_BYTES  = FB_PIXELS * sizeof(uint16_t);
    uint16_t* fb[2] = {nullptr, nullptr};
    for (int i=0; i<2; ++i) {
        fb[i] = (uint16_t*) heap_caps_malloc(FB_BYTES, MALLOC_CAP_DMA | MALLOC_CAP_INTERNAL);
        if (!fb[i]) fb[i] = (uint16_t*) heap_caps_malloc(FB_BYTES, MALLOC_CAP_DMA | MALLOC_CAP_SPIRAM);
        configASSERT(fb[i]);
    }
    int cur = 0;

    // 7) Precompute scale maps (nearest neighbor)
    static int xmap[LCD_W];
    static int ymap[LCD_H];
    for (int x=0; x<LCD_W; ++x)  xmap[x] = (int)((int64_t)x * src_w / LCD_W);
    for (int y=0; y<LCD_H; ++y)  ymap[y] = (int)((int64_t)y * src_h / LCD_H);

    ESP_LOGI(TAG, "Start capture/display loop: src=%dx%d %s -> LCD=240x240",
             src_w, src_h, (fourcc==V4L2_PIX_FMT_RGB565) ? "RGB565" : (fourcc==V4L2_PIX_FMT_YUYV ? "YUYV" : "OTHER"));

    while (true) {
        // ---- Dequeue ----
        v4l2_buffer buf{};
        buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;

        if (ioctl(fd, VIDIOC_DQBUF, &buf) != 0) {
            ESP_LOGW(TAG, "DQBUF failed, retry");
            vTaskDelay(pdMS_TO_TICKS(1));
            continue;
        }
        if (!(buf.flags & V4L2_BUF_FLAG_DONE)) {
            ioctl(fd, VIDIOC_QBUF, &buf);
            continue;
        }

        const uint8_t* frame = (const uint8_t*)buffers[buf.index].start;
        uint16_t* dst = fb[cur];

        if (cam_rgb565 && src_w == LCD_W && src_h == LCD_H) {
            // Snel pad: 240x240 RGB565 -> byteswap naar MSB-first
            const uint16_t* src16 = (const uint16_t*)frame;
            for (size_t i=0; i<FB_PIXELS; ++i) dst[i] = bswap16(src16[i]);
        } else if (cam_rgb565) {
            // Schalen RGB565 -> 240x240
            const uint16_t* src16 = (const uint16_t*)frame;
            for (int dy=0; dy<LCD_H; ++dy) {
                int sy = ymap[dy];
                const uint16_t* src_line = src16 + (size_t)sy * src_w;
                uint16_t*       out_line = dst + (size_t)dy * LCD_W;
                for (int dx=0; dx<LCD_W; ++dx) {
                    int sx = xmap[dx];
                    out_line[dx] = bswap16(src_line[sx]);
                }
            }
        } else if (cam_yuyv) {
            // Schalen YUYV -> 240x240
            for (int dy=0; dy<LCD_H; ++dy) {
                int sy = ymap[dy];
                const uint8_t* src_line = frame + (size_t)sy * (size_t)src_w * 2;
                uint16_t* out_line = dst + (size_t)dy * LCD_W;
                for (int dx=0; dx<LCD_W; ++dx) {
                    int sx = xmap[dx] & ~1; // per 2 pixels
                    out_line[dx] = bswap16(yuyv_at_to_rgb565(src_line, src_w, sx));
                }
            }
        }

        // ---- Volledige frame flush (queue depth = 1) ----
        ESP_ERROR_CHECK(esp_lcd_panel_draw_bitmap(panel, 0, 0, LCD_W, LCD_H, dst));

        // ---- Requeue capture buffer ----
        ESP_ERROR_CHECK(ioctl(fd, VIDIOC_QBUF, &buf));

        // wissel framebuffer
        cur ^= 1;
    }
}
