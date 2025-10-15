#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "driver/spi_master.h"
#include "driver/gpio.h"
#include "esp_lcd_panel_io.h"
#include "esp_lcd_panel_vendor.h"
#include "esp_lcd_panel_ops.h"
#include "esp_lcd_panel_st7789.h"

static const char* TAG = "ILI9341_P4";

#define LCD_HOST        SPI2_HOST
#define PIN_LCD_SCLK    46
#define PIN_LCD_MOSI    47
#define PIN_LCD_CS      45
#define PIN_LCD_DC      48
#define PIN_LCD_RST     6

#define LCD_HRES        240
#define LCD_VRES        320

extern "C" void app_main(void) {
    ESP_LOGI(TAG, "Init SPI bus");
    spi_bus_config_t buscfg = {
        .mosi_io_num = PIN_LCD_MOSI,
        .miso_io_num = -1,
        .sclk_io_num = PIN_LCD_SCLK,
        .quadwp_io_num = -1,
        .quadhd_io_num = -1,
        .max_transfer_sz = LCD_HRES * 40 * 2,
        .flags = SPICOMMON_BUSFLAG_MASTER
    };
    ESP_ERROR_CHECK(spi_bus_initialize(LCD_HOST, &buscfg, SPI_DMA_CH_AUTO));

    ESP_LOGI(TAG, "Init panel IO (SPI)");
    esp_lcd_panel_io_handle_t io_handle = nullptr;
    esp_lcd_panel_io_spi_config_t io_config = {
        .cs_gpio_num = PIN_LCD_CS,
        .dc_gpio_num = PIN_LCD_DC,
        .spi_mode = 0,
        .pclk_hz = 40 * 1000 * 1000,
        .trans_queue_depth = 10,
        .lcd_cmd_bits = 8,
        .lcd_param_bits = 8,
    };
    ESP_ERROR_CHECK(esp_lcd_new_panel_io_spi(LCD_HOST, &io_config, &io_handle));

    ESP_LOGI(TAG, "Create ILI9341 panel");
    esp_lcd_panel_handle_t panel = nullptr;
    esp_lcd_panel_dev_config_t panel_config = {
        .reset_gpio_num = PIN_LCD_RST,
        .color_space = ESP_LCD_COLOR_SPACE_RGB,
        .bits_per_pixel = 16,
    };
    ESP_ERROR_CHECK(esp_lcd_new_panel_st7789(io_handle, &panel_config, &panel));

    ESP_ERROR_CHECK(esp_lcd_panel_reset(panel));
    ESP_ERROR_CHECK(esp_lcd_panel_init(panel));

    ESP_ERROR_CHECK(esp_lcd_panel_swap_xy(panel, false));
    ESP_ERROR_CHECK(esp_lcd_panel_mirror(panel, false, false));
    ESP_ERROR_CHECK(esp_lcd_panel_set_gap(panel, 0, 0));

    
    uint16_t blue = (31 << 11);
    const int w = LCD_HRES, h = LCD_VRES;
    size_t buf_px = w * 20;
    uint16_t *line = (uint16_t*)heap_caps_malloc(buf_px * sizeof(uint16_t), MALLOC_CAP_DMA);
    for (size_t i = 0; i < buf_px; ++i) line[i] = blue;

    for (int y = 0; y < h; y += 20) {
        int y2 = (y + 20 > h) ? h : (y + 20);
        ESP_ERROR_CHECK(esp_lcd_panel_draw_bitmap(panel, 0, y, w, y2, line));
    }
    heap_caps_free(line);

    ESP_LOGI(TAG, "Done. Looping...");
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(1000));
        ESP_LOGI(TAG, "tick");
    }
}