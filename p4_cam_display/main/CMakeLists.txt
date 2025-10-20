idf_component_register(
  SRCS "cam_display.cpp"
  INCLUDE_DIRS "."
  REQUIRES
    driver
    esp_lcd
    esp_lcd_gc9a01
    esp_timer
    heap
    freertos
    esp_system
    example_video_common   # <— dit is het mapje uit je zip (als component)
    esp_hw_support
)

target_compile_options(${COMPONENT_LIB} PRIVATE -std=gnu++17)
