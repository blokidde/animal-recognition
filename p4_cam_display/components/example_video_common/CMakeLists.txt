set(srcs "example_init_video.c")
set(inc_dirs "include")

if(NOT CONFIG_IDF_TARGET_ESP32C61)
    list(APPEND srcs "example_encoder.c")
endif()

# if(CONFIG_EXAMPLE_SELECT_ESP32P4_FUNCTION_EV_BOARD_V1_4)
#     list(APPEND inc_dirs "include/boards/esp32-p4-function-ev-board-v1.4")
# elseif(CONFIG_EXAMPLE_SELECT_ESP32P4_FUNCTION_EV_BOARD_V1_5)
#     list(APPEND inc_dirs "include/boards/esp32-p4-function-ev-board-v1.5")
# elseif(CONFIG_EXAMPLE_SELECT_ESP32P4_EYE)
#     list(APPEND inc_dirs "include/boards/esp32-p4-eye")
# elseif(CONFIG_EXAMPLE_SELECT_ESP32S3_EYE)
#     list(APPEND inc_dirs "include/boards/esp32-s3-eye")
# elseif(CONFIG_EXAMPLE_SELECT_CUSTOMIZED_DEV_BOARD)
#     list(APPEND inc_dirs "include/boards/customized")
# endif()

idf_component_register(
  SRCS
    "example_init_video.c"
    "example_encoder.c"
  INCLUDE_DIRS
    "include"
  REQUIRES
    esp_video         # had je al/komt uit de registry
    esp_new_jpeg      # software JPEG lib (komt al binnen)
    esp_driver_jpeg   # <-- nodig voor driver/jpeg_encode.h
    driver            # (gpio/i2c/spi, meestal toch al aanwezig)
)

# in components/example_video_common/CMakeLists.txt
if(CONFIG_EXAMPLE_SELECT_CUSTOMIZED_DEV_BOARD)
  target_include_directories(${COMPONENT_LIB} PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include/boards/customized"
  )
endif()



if(CONFIG_EXAMPLE_SELECT_JPEG_HW_DRIVER)
    idf_component_optional_requires(PRIVATE "esp_driver_jpeg")
endif()
