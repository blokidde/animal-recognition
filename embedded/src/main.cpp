#include <stdio.h>
extern "C" {
  #include "freertos/FreeRTOS.h"
  #include "freertos/task.h"
  #include "driver/spi_master.h"
  #include "driver/gpio.h"
  #include "esp_log.h"
  #include "esp_lcd_panel_ops.h"
  #include "esp_lcd_io_spi.h"
  #include "esp_lcd_ili9341.h"
}

#define TAG "ILI9341_P4"

// ---- Pins (pas aan jouw wiring aan) ----
static constexpr spi_host_device_t LCD_HOST = SPI2_HOST;

// Gebruik int voor buscfg (akkoord met IDF) en cast naar gpio_num_t waar nodig.
static constexpr int PIN_LCD_SCLK = 32;
static constexpr int PIN_LCD_MOSI = 26;
static constexpr int PIN_LCD_MISO = -1;    // write-only display
static constexpr int PIN_LCD_CS   = 25;
static constexpr int PIN_LCD_DC   = 27;
static constexpr int PIN_LCD_RST  = 20;    // zet op -1 als je geen RST-draad hebt
static constexpr int PIN_LCD_BL   = 21;    // zet op -1 als je geen backlight aanstuurt

// ---- Display config ----
static constexpr int LCD_HRES = 240;
static constexpr int LCD_VRES = 320;
static constexpr uint32_t LCD_SPI_HZ = (40u * 1000u * 1000u);  // 40 MHz

// Helper: int -> gpio_num_t, met NC-guard
static inline gpio_num_t to_gpio(int pin) {
  return (pin >= 0) ? static_cast<gpio_num_t>(pin) : GPIO_NUM_NC; // -1 -> NC
}

extern "C" void app_main(void)
{
  ESP_LOGI(TAG, "Init SPI bus");

  spi_bus_config_t buscfg{};
  // In IDF zijn dit 'int' velden; we casten expliciet naar int voor duidelijkheid.
  buscfg.sclk_io_num     = static_cast<int>(PIN_LCD_SCLK);
  buscfg.mosi_io_num     = static_cast<int>(PIN_LCD_MOSI);
  buscfg.miso_io_num     = static_cast<int>(PIN_LCD_MISO);
  buscfg.quadwp_io_num   = -1;
  buscfg.quadhd_io_num   = -1;
  buscfg.max_transfer_sz = LCD_HRES * 80 * static_cast<int>(sizeof(uint16_t)); // 80 lijnen

  ESP_ERROR_CHECK(spi_bus_initialize(LCD_HOST, &buscfg, SPI_DMA_CH_AUTO));

  // Backlight (optioneel)
  if (PIN_LCD_BL >= 0) {
    gpio_config_t io_conf{};
    io_conf.pin_bit_mask = (1ULL << static_cast<uint64_t>(PIN_LCD_BL));
    io_conf.mode = GPIO_MODE_OUTPUT;
    io_conf.pull_up_en = GPIO_PULLUP_DISABLE;
    io_conf.pull_down_en = GPIO_PULLDOWN_DISABLE;
    gpio_config(&io_conf);
    ESP_ERROR_CHECK(gpio_set_level(to_gpio(PIN_LCD_BL), 1));
  }

  ESP_LOGI(TAG, "Install panel IO (SPI)");
  esp_lcd_panel_io_handle_t io_handle = nullptr;

  esp_lcd_panel_io_spi_config_t io_cfg{};
  // Deze velden zijn in nieuwere IDF's gpio_num_t → cast expliciet.
  io_cfg.dc_gpio_num       = to_gpio(PIN_LCD_DC);
  io_cfg.cs_gpio_num       = to_gpio(PIN_LCD_CS);
  io_cfg.pclk_hz           = LCD_SPI_HZ;
  io_cfg.lcd_cmd_bits      = 8;
  io_cfg.lcd_param_bits    = 8;
  io_cfg.spi_mode          = 0;
  io_cfg.trans_queue_depth = 10;

  ESP_ERROR_CHECK(esp_lcd_new_panel_io_spi(
      (esp_lcd_spi_bus_handle_t)LCD_HOST, &io_cfg, &io_handle));

  ESP_LOGI(TAG, "Install ILI9341 panel driver");
  esp_lcd_panel_handle_t panel = nullptr;

  esp_lcd_panel_dev_config_t panel_cfg{};
  panel_cfg.reset_gpio_num = to_gpio(PIN_LCD_RST);         // GPIO_NUM_NC als -1
  panel_cfg.rgb_ele_order  = LCD_RGB_ELEMENT_ORDER_BGR;    // veel ILI9341's zijn BGR
  panel_cfg.bits_per_pixel = 16;                           // RGB565

  ESP_ERROR_CHECK(esp_lcd_new_panel_ili9341(io_handle, &panel_cfg, &panel));

  // Reset + init + display aan
  ESP_ERROR_CHECK(esp_lcd_panel_reset(panel));
  ESP_ERROR_CHECK(esp_lcd_panel_init(panel));
  ESP_ERROR_CHECK(esp_lcd_panel_disp_on_off(panel, true));

  // Eenvoudige test: vul scherm rood
  static uint16_t line[LCD_HRES];
  for (int x = 0; x < LCD_HRES; ++x) line[x] = 0xF800; // RGB565 rood
  for (int y = 0; y < LCD_VRES; ++y) {
    ESP_ERROR_CHECK(esp_lcd_panel_draw_bitmap(panel, 0, y, LCD_HRES, y + 1, line));
  }

  ESP_LOGI(TAG, "Rood scherm getekend!");
  while (true) {
    vTaskDelay(pdMS_TO_TICKS(1000));
    ESP_LOGI(TAG, "tick");
  }
}
