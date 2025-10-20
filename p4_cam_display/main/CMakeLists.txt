idf_component_register(
  SRCS "cam_display.cpp"
  INCLUDE_DIRS "."
  REQUIRES
    driver          # spi_master, gpio
    esp_lcd         # lcd driver + panel API
    esp_timer       # esp_timer_get_time()
    heap            # heap_caps_malloc
    freertos        # FreeRTOS kernel
    esp_system      # bevat nu log-functionaliteit
)
target_compile_options(${COMPONENT_LIB} PRIVATE -std=gnu++17)
