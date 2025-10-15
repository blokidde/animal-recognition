#include <stdio.h>
#include <string.h>
#include <vector>

extern "C" {
  #include "freertos/FreeRTOS.h"
  #include "freertos/task.h"
  #include "driver/spi_master.h"
  #include "driver/gpio.h"
  #include "esp_log.h"
  #include "esp_timer.h"
  #include "esp_lcd_panel_ops.h"
  #include "esp_lcd_io_spi.h"
  #include "esp_lcd_ili9341.h"
}

#define TAG "ILI9341_P4"

// ---- Pins ----
static constexpr spi_host_device_t LCD_HOST = SPI2_HOST;
static constexpr int PIN_LCD_SCLK = 32;
static constexpr int PIN_LCD_MOSI = 26;
static constexpr int PIN_LCD_MISO = -1;    // write-only
static constexpr int PIN_LCD_CS   = 25;
static constexpr int PIN_LCD_DC   = 27;
static constexpr int PIN_LCD_RST  = 20;    // -1 als geen RST
static constexpr int PIN_LCD_BL   = 21;    // -1 als geen backlight

// ---- Panel resolutie (fysiek) ----
static constexpr int LCD_HRES = 240;
static constexpr int LCD_VRES = 320;
static constexpr uint32_t LCD_SPI_HZ = (40u * 1000u * 1000u);  // 40 MHz

// ---- Tekst/timing ----
static constexpr int SCALE      = 7;   // iets kleiner dan voorheen
static constexpr int REFRESH_MS = 33;  // ~30 FPS

// ---- Oriëntatie voor LIGGEND ----
// We draaien naar landscape met swap_xy = true.
// Deze combinatie geeft vaak "liggend, juiste leesrichting":
static constexpr bool ORIENT_SWAP_XY  = true;
static constexpr bool ORIENT_MIRROR_X = true;
static constexpr bool ORIENT_MIRROR_Y = true;

// Helpers
static inline gpio_num_t to_gpio(int pin) {
  return (pin >= 0) ? static_cast<gpio_num_t>(pin) : GPIO_NUM_NC;
}
static inline uint16_t RGB565(uint8_t r, uint8_t g, uint8_t b) {
  return (uint16_t)(((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3));
}
static constexpr uint16_t COLOR_BLACK = 0x0000;
static constexpr uint16_t COLOR_WHITE = 0xFFFF;

// 5x7 font (digits + '.' + ',')
static const uint8_t FONT5x7_DIGIT[10][5] = {
  {0x3E,0x51,0x49,0x45,0x3E}, {0x00,0x42,0x7F,0x40,0x00},
  {0x42,0x61,0x51,0x49,0x46}, {0x21,0x41,0x45,0x4B,0x31},
  {0x18,0x14,0x12,0x7F,0x10}, {0x27,0x45,0x45,0x45,0x39},
  {0x3C,0x4A,0x49,0x49,0x30}, {0x01,0x71,0x09,0x05,0x03},
  {0x36,0x49,0x49,0x49,0x36}, {0x06,0x49,0x49,0x29,0x1E}
};
static const uint8_t FONT5x7_DOT[1]   = { 0x40 };        // '.'
static const uint8_t FONT5x7_COMMA[2] = { 0x40, 0x20 };  // ','

static inline int char_width(char c, int scale) {
  if (c >= '0' && c <= '9') return (5 + 1) * scale;
  if (c == '.')             return (1 + 1) * scale;
  if (c == ',')             return (2 + 1) * scale;
  if (c == ' ')             return (3) * scale;
  return 0;
}
static int measure_text_px(const char* s, int scale) {
  int w = 0; for (const char* p = s; *p; ++p) w += char_width(*p, scale); return w;
}

// Render tekst in 1 buffer (bg gevuld), daarna één draw_bitmap → geen flikker
static void render_text_to_buffer(const char* s, int scale,
                                  uint16_t fg, uint16_t bg,
                                  std::vector<uint16_t>& buf, int& out_w, int& out_h)
{
  out_h = 7 * scale;
  out_w = measure_text_px(s, scale);
  if (out_w <= 0) { buf.clear(); return; }
  buf.assign(out_w * out_h, bg);

  int cx = 0;
  for (const char* p = s; *p; ++p) {
    if (*p >= '0' && *p <= '9') {
      const uint8_t* cols = FONT5x7_DIGIT[*p - '0'];
      for (int sx = 0; sx < 5; ++sx)
        for (int sy = 0; sy < 7; ++sy)
          if ((cols[sx] >> sy) & 0x01)
            for (int dx = 0; dx < scale; ++dx)
              for (int dy = 0; dy < scale; ++dy)
                buf[(sy*scale + dy) * out_w + (cx + sx*scale + dx)] = fg;
      cx += (5 + 1) * scale;
    } else if (*p == '.') {
      for (int dy = 0; dy < scale; ++dy)
        buf[((6*scale) + dy) * out_w + cx] = fg;
      cx += (1 + 1) * scale;
    } else if (*p == ',') {
      for (int dy = 0; dy < scale; ++dy) buf[((5*scale)+dy) * out_w + cx] = fg;
      for (int dy = 0; dy < scale; ++dy) buf[((6*scale)+dy) * out_w + (cx+scale)] = fg;
      cx += (2 + 1) * scale;
    } else if (*p == ' ') {
      cx += (3) * scale;
    }
  }
}

extern "C" void app_main(void)
{
  // SPI bus
  spi_bus_config_t buscfg{};
  buscfg.sclk_io_num     = PIN_LCD_SCLK;
  buscfg.mosi_io_num     = PIN_LCD_MOSI;
  buscfg.miso_io_num     = PIN_LCD_MISO;
  buscfg.quadwp_io_num   = -1;
  buscfg.quadhd_io_num   = -1;
  buscfg.max_transfer_sz = LCD_HRES * 80 * (int)sizeof(uint16_t);
  ESP_ERROR_CHECK(spi_bus_initialize(LCD_HOST, &buscfg, SPI_DMA_CH_AUTO));

  // Backlight
  if (PIN_LCD_BL >= 0) {
    gpio_config_t io_conf{};
    io_conf.pin_bit_mask = (1ULL << (uint64_t)PIN_LCD_BL);
    io_conf.mode = GPIO_MODE_OUTPUT;
    gpio_config(&io_conf);
    gpio_set_level(to_gpio(PIN_LCD_BL), 1);
  }

  // IO + panel
  esp_lcd_panel_io_handle_t io_handle = nullptr;
  esp_lcd_panel_io_spi_config_t io_cfg{};
  io_cfg.dc_gpio_num       = to_gpio(PIN_LCD_DC);
  io_cfg.cs_gpio_num       = to_gpio(PIN_LCD_CS);
  io_cfg.pclk_hz           = LCD_SPI_HZ;
  io_cfg.lcd_cmd_bits      = 8;
  io_cfg.lcd_param_bits    = 8;
  io_cfg.spi_mode          = 0;
  io_cfg.trans_queue_depth = 10;
  ESP_ERROR_CHECK(esp_lcd_new_panel_io_spi((esp_lcd_spi_bus_handle_t)LCD_HOST, &io_cfg, &io_handle));

  esp_lcd_panel_handle_t panel = nullptr;
  esp_lcd_panel_dev_config_t panel_cfg{};
  panel_cfg.reset_gpio_num = to_gpio(PIN_LCD_RST);
  panel_cfg.rgb_ele_order  = LCD_RGB_ELEMENT_ORDER_RGB; // 0xF800 = rood
  panel_cfg.bits_per_pixel = 16;                        // RGB565
  ESP_ERROR_CHECK(esp_lcd_new_panel_ili9341(io_handle, &panel_cfg, &panel));

  // Init + aan
  ESP_ERROR_CHECK(esp_lcd_panel_reset(panel));
  ESP_ERROR_CHECK(esp_lcd_panel_init(panel));
  ESP_ERROR_CHECK(esp_lcd_panel_disp_on_off(panel, true));

  // ---- LIGGEND instellen ----
  ESP_ERROR_CHECK(esp_lcd_panel_swap_xy(panel, ORIENT_SWAP_XY));
  ESP_ERROR_CHECK(esp_lcd_panel_mirror(panel, ORIENT_MIRROR_X, ORIENT_MIRROR_Y));
  // Als de tekst tóch gespiegeld is:
  // 1) zet ORIENT_MIRROR_X = false, ORIENT_MIRROR_Y = true
  // 2) of laat ORIENT_SWAP_XY = true maar wissel de mirrors om

  // Logische schermmaat (na swap_xy wisselen W/H)
  const int SCREEN_W = ORIENT_SWAP_XY ? LCD_VRES : LCD_HRES; // 320
  const int SCREEN_H = ORIENT_SWAP_XY ? LCD_HRES : LCD_VRES; // 240

  // Clear screen (in stroken)
  std::vector<uint16_t> clear(SCREEN_W * 16, COLOR_BLACK);
  for (int y = 0; y < SCREEN_H; y += 16) {
    int h = (y + 16 <= SCREEN_H) ? 16 : (SCREEN_H - y);
    esp_lcd_panel_draw_bitmap(panel, 0, y, SCREEN_W, y + h, clear.data());
  }

  // Timer
  int64_t t0_us = esp_timer_get_time();
  std::vector<uint16_t> buf;
  char txt[32];
  int last_x = 0, last_y = 0, last_w = 0, last_h = 0;

  while (true) {
    // Nauwkeurige tijd, 2 decimalen en komma
    double secs = (esp_timer_get_time() - t0_us) / 1e6;
    snprintf(txt, sizeof(txt), "%.2f", secs);
    for (char* p = txt; *p; ++p) if (*p == '.') *p = ',';

    // Render één buffer & draw in 1 call
    int w = 0, h = 0;
    render_text_to_buffer(txt, SCALE, COLOR_WHITE, COLOR_BLACK, buf, w, h);
    if (w == 0 || h == 0) { vTaskDelay(pdMS_TO_TICKS(REFRESH_MS)); continue; }

    // Centreer in LIGGEND scherm
    int x = (SCREEN_W - w) / 2;
    int y = (SCREEN_H - h) / 2;

    // Vorige tekst overschrijven (zwart), dan nieuwe
    if (last_w > 0 && last_h > 0) {
      std::vector<uint16_t> black(last_w * last_h, COLOR_BLACK);
      esp_lcd_panel_draw_bitmap(panel, last_x, last_y, last_x + last_w, last_y + last_h, black.data());
    }
    esp_lcd_panel_draw_bitmap(panel, x, y, x + w, y + h, buf.data());

    last_x = x; last_y = y; last_w = w; last_h = h;

    vTaskDelay(pdMS_TO_TICKS(REFRESH_MS));
  }
}
