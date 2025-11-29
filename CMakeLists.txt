cmake_minimum_required(VERSION 3.5)

# Set extension name here
set(TARGET_NAME yardstick)

set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

project(${TARGET_NAME})
include_directories(src/include)

# Path to yardstick-rs
set(YARDSTICK_RS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/yardstick-rs")
set(YARDSTICK_LIB "${YARDSTICK_RS_DIR}/target/release/libyardstick.a")
set(YARDSTICK_INCLUDE "${YARDSTICK_RS_DIR}/include")

# Include Rust library headers
include_directories(${YARDSTICK_INCLUDE})

set(EXTENSION_SOURCES src/yardstick_extension.cpp)

build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

# Mark Rust library as imported so CMake tracks it
add_library(yardstick_rust STATIC IMPORTED)
set_target_properties(yardstick_rust PROPERTIES IMPORTED_LOCATION ${YARDSTICK_LIB})

# Link the Rust static library
target_link_libraries(${EXTENSION_NAME} yardstick_rust)
target_link_libraries(${LOADABLE_EXTENSION_NAME} yardstick_rust)

# On macOS, we need to link against some system frameworks
if(APPLE)
    target_link_libraries(${EXTENSION_NAME} "-framework Security" "-framework CoreFoundation")
    target_link_libraries(${LOADABLE_EXTENSION_NAME} "-framework Security" "-framework CoreFoundation")
endif()

# On Linux, we need pthread and dl
if(UNIX AND NOT APPLE)
    target_link_libraries(${EXTENSION_NAME} pthread dl)
    target_link_libraries(${LOADABLE_EXTENSION_NAME} pthread dl)
endif()

install(
  TARGETS ${EXTENSION_NAME}
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
