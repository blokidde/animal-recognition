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
    example_video_common
    esp_hw_support
    esp_driver_isp
)

target_compile_features(${COMPONENT_LIB} PRIVATE cxx_std_20)
