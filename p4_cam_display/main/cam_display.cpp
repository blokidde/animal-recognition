// cam_display.cpp
#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/param.h>
#include <unistd.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"

#include "driver/spi_master.h"
#include "driver/gpio.h"

#include "esp_lcd_panel_ops.h"
#include "esp_lcd_io_spi.h"
#include "esp_lcd_gc9a01.h"

// -------- ESP-Video / V4L2 ----------
#include "example_video_common.h"     // zet /dev/* aanmaak op basis van Kconfig
#include <linux/videodev2.h>

static const char* TAG = "P4_CAM_GC9A01";

// ==== (optioneel) board-level power enable ====
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
static constexpr uint32_t LCD_SPI_HZ = (60u * 1000u * 1000u);  // 60 MHz

// Oriëntatie
static constexpr bool ORIENT_SWAP_XY  = false;
static constexpr bool ORIENT_MIRROR_X = true;
static constexpr bool ORIENT_MIRROR_Y = false;

// Stripe hoogte (DMA-vriendelijk)
static constexpr int STRIPE_H = 80;

// ---- Helpers ----
static inline gpio_num_t to_gpio(int pin) { return (pin >= 0) ? static_cast<gpio_num_t>(pin) : GPIO_NUM_NC; }
static inline uint16_t pack_rgb565(int r, int g, int b) {
    uint16_t R = (uint16_t)((r & 0xF8) << 8);
    uint16_t G = (uint16_t)((g & 0xFC) << 3);
    uint16_t B = (uint16_t)((b >> 3) & 0x1F);
    return (uint16_t)(R | G | B);
}
static inline int clip8(int x){ return x < 0 ? 0 : (x > 255 ? 255 : x); }

// YUYV (YUV422) → RGB565
static inline uint16_t yuyv_pixel_to_rgb565(const uint8_t* line, int src_w, int x) {
    int pair_x = x & ~1;
    const uint8_t* p = line + pair_x*2; // [Y0 U Y1 V]
    int Y = p[(x & 1) ? 2 : 0];
    int U = p[1];
    int V = p[3];

    // BT.601
    int C = Y - 16;
    int D = U - 128;
    int E = V - 128;
    int r = clip8((298 * C + 409 * E + 128) >> 8);
    int g = clip8((298 * C - 100 * D - 208 * E + 128) >> 8);
    int b = clip8((298 * C + 516 * D + 128) >> 8);
    return pack_rgb565(r, g, b);
}

static inline uint16_t rgb565_sample(const uint16_t* base, int src_w, int x, int y){
    return base[y * src_w + x];
}

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

// ==== (optioneel) voeding aanzetten via GPIO ====
static void maybe_enable_cam_power_gpio(void)
{
    if (CAM_PWR_EN_GPIO >= 0) {
        gpio_config_t io{};
        io.pin_bit_mask = (1ULL << (unsigned)CAM_PWR_EN_GPIO);
        io.mode = GPIO_MODE_OUTPUT;
        io.pull_down_en = GPIO_PULLDOWN_DISABLE;
        io.pull_up_en   = GPIO_PULLUP_DISABLE;
        gpio_config(&io);
        gpio_set_level(to_gpio(CAM_PWR_EN_GPIO), 1);
        ESP_LOGI(TAG, "CAM_PWR_EN GPIO %d set HIGH", CAM_PWR_EN_GPIO);
        vTaskDelay(pdMS_TO_TICKS(5));
    } else {
        ESP_LOGI(TAG, "Geen CAM_PWR_EN GPIO ingesteld (overslaan).");
    }
}

extern "C" void app_main(void)
{
    // 0) (optioneel) power-enable
    maybe_enable_cam_power_gpio();

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

    // 3) Query formaat & buffers
    v4l2_format fmt{};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ESP_ERROR_CHECK(ioctl(fd, VIDIOC_G_FMT, &fmt));

    const uint32_t src_w = fmt.fmt.pix.width;
    const uint32_t src_h = fmt.fmt.pix.height;
    const uint32_t fourcc = fmt.fmt.pix.pixelformat;

    ESP_LOGI(TAG, "Camera fmt: %ux%u fourcc=0x%08X", src_w, src_h, fourcc);

    const bool is_yuyv   = (fourcc == V4L2_PIX_FMT_YUYV);
    const bool is_rgb565 = (fourcc == V4L2_PIX_FMT_RGB565);
    if (!is_yuyv && !is_rgb565) {
        ESP_LOGE(TAG, "Unsupported pixel format (need YUYV or RGB565). Pas sensor/driver output aan.");
        vTaskDelay(portMAX_DELAY);
    }

    // Buffer request
    v4l2_requestbuffers req{};
    req.count  = 4;
    req.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    ESP_ERROR_CHECK(ioctl(fd, VIDIOC_REQBUFS, &req));
    if (req.count < 2) {
        ESP_LOGE(TAG, "Onvoldoende v4l2 buffers");
        vTaskDelay(portMAX_DELAY);
    }

    // mmap buffers
    struct MMapBuf { void* start; size_t length; } buffers[8];
    memset(buffers, 0, sizeof(buffers));
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

    // STREAMON
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ESP_ERROR_CHECK(ioctl(fd, VIDIOC_STREAMON, &type));

    // 4) Init LCD (GC9A01)
    spi_bus_config_t buscfg{};
    buscfg.sclk_io_num     = PIN_LCD_SCLK;
    buscfg.mosi_io_num     = PIN_LCD_MOSI;
    buscfg.miso_io_num     = PIN_LCD_MISO;
    buscfg.quadwp_io_num   = -1;
    buscfg.quadhd_io_num   = -1;
    buscfg.max_transfer_sz = LCD_W * STRIPE_H * (int)sizeof(uint16_t);
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
    io_cfg.trans_queue_depth = 16;
    ESP_ERROR_CHECK(esp_lcd_new_panel_io_spi((esp_lcd_spi_bus_handle_t)LCD_HOST, &io_cfg, &io_handle));

    esp_lcd_panel_handle_t panel = nullptr;
    esp_lcd_panel_dev_config_t panel_cfg{};
    panel_cfg.reset_gpio_num = to_gpio(PIN_LCD_RST);
    panel_cfg.rgb_ele_order  = LCD_RGB_ELEMENT_ORDER_BGR;   // <<— GC9A01 modules gebruiken vaak BGR
    panel_cfg.bits_per_pixel = 16; // RGB565
    ESP_ERROR_CHECK(esp_lcd_new_panel_gc9a01(io_handle, &panel_cfg, &panel));

    ESP_ERROR_CHECK(esp_lcd_panel_reset(panel));
    vTaskDelay(pdMS_TO_TICKS(20));
    ESP_ERROR_CHECK(esp_lcd_panel_init(panel));
    vTaskDelay(pdMS_TO_TICKS(120));
    ESP_ERROR_CHECK(esp_lcd_panel_disp_on_off(panel, true));

    // Invert ON is vaak nodig bij GC9A01 breakout boards
    ESP_ERROR_CHECK(esp_lcd_panel_invert_color(panel, true));

    ESP_ERROR_CHECK(esp_lcd_panel_swap_xy(panel, ORIENT_SWAP_XY));
    ESP_ERROR_CHECK(esp_lcd_panel_mirror(panel, ORIENT_MIRROR_X, ORIENT_MIRROR_Y));
    ESP_ERROR_CHECK(esp_lcd_panel_set_gap(panel, 0, 0)); // expliciet 0 offset

    // 5) Stripe buffer (RGB565)
    size_t stripe_bytes = (size_t)LCD_W * STRIPE_H * sizeof(uint16_t);
    uint16_t* stripe = (uint16_t*) heap_caps_malloc(stripe_bytes, MALLOC_CAP_DMA | MALLOC_CAP_INTERNAL);
    if (!stripe) stripe = (uint16_t*) heap_caps_malloc(stripe_bytes, MALLOC_CAP_DMA | MALLOC_CAP_SPIRAM);
    configASSERT(stripe);

    // Crop naar square
    const uint32_t S = (src_w < src_h) ? src_w : src_h;
    const uint32_t crop_x = (src_w - S) / 2;
    const uint32_t crop_y = (src_h - S) / 2;
    ESP_LOGI(TAG, "Crop square: %ux%u @ (%u,%u) -> %dx%d", S, S, crop_x, crop_y, LCD_W, LCD_H);

    // 6) Capture → Convert → Push
    while (true) {
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

        for (int dst_y0 = 0; dst_y0 < LCD_H; dst_y0 += STRIPE_H) {
            const int h = MIN(STRIPE_H, LCD_H - dst_y0);
            for (int dy = 0; dy < h; ++dy) {
                int dst_y = dst_y0 + dy;
                uint32_t src_y = crop_y + (uint32_t)((uint64_t)dst_y * S / LCD_H);

                const uint8_t*    src_line_u8     = frame + (size_t)src_y * (size_t)src_w * 2;
                const uint16_t*   src_line_rgb565 = (const uint16_t*)(frame + (size_t)src_y * (size_t)src_w * 2);

                uint16_t* out = stripe + dy * LCD_W;

                for (int dst_x = 0; dst_x < LCD_W; ++dst_x) {
                    uint32_t src_x = crop_x + (uint32_t)((uint64_t)dst_x * S / LCD_W);

                    uint16_t pix565 = is_yuyv
                        ? yuyv_pixel_to_rgb565(src_line_u8, src_w, (int)src_x)
                        : rgb565_sample(src_line_rgb565, src_w, (int)src_x, 0);

                    // --- Belangrijk: byteswappen naar MSB-first voor SPI ---
                    out[dst_x] = __builtin_bswap16(pix565);
                }
            }

            ESP_ERROR_CHECK(esp_lcd_panel_draw_bitmap(panel,
                                                      0, dst_y0,
                                                      LCD_W, dst_y0 + h,
                                                      stripe));
        }

        ioctl(fd, VIDIOC_QBUF, &buf);
    }
}
