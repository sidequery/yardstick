cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED 1)

# Set extension name here
set(TARGET_NAME yardstick)

set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

project(${TARGET_NAME})
include_directories(src/include)
include_directories(include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/duckdb/src/include)

# Detect Rust target based on platform
execute_process(
    COMMAND rustup target list --installed
    OUTPUT_VARIABLE RUST_TARGETS
)

set(Rust_CARGO_TARGET "")
if("${OS_NAME}" STREQUAL "linux")
    if ("${OS_ARCH}" STREQUAL "arm64")
        set(Rust_CARGO_TARGET "aarch64-unknown-linux-gnu")
    elseif("${CMAKE_CXX_COMPILER}" MATCHES "aarch64")
        set(Rust_CARGO_TARGET "aarch64-unknown-linux-gnu")
    else()
        string(FIND "${RUST_TARGETS}" "musl" MUSL_TARGET_FOUND)
        if(NOT MUSL_TARGET_FOUND EQUAL -1)
            set(Rust_CARGO_TARGET "x86_64-unknown-linux-musl")
        else()
            set(Rust_CARGO_TARGET "x86_64-unknown-linux-gnu")
        endif()
    endif()
elseif("${OS_NAME}" STREQUAL "osx" OR APPLE)
    if ("${OSX_BUILD_ARCH}" STREQUAL "arm64")
        set(Rust_CARGO_TARGET "aarch64-apple-darwin")
    elseif ("${OSX_BUILD_ARCH}" STREQUAL "x86_64")
        set(Rust_CARGO_TARGET "x86_64-apple-darwin")
    elseif ("${OS_ARCH}" STREQUAL "arm64")
        set(Rust_CARGO_TARGET "aarch64-apple-darwin")
    elseif ("${OS_ARCH}" STREQUAL "amd64")
        set(Rust_CARGO_TARGET "x86_64-apple-darwin")
    elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
        set(Rust_CARGO_TARGET "aarch64-apple-darwin")
    else()
        # Default to x86_64 for macOS
        set(Rust_CARGO_TARGET "x86_64-apple-darwin")
    endif()
elseif(WIN32)
   # Determine Windows architecture - OS_ARCH may not be set when running CMake directly
   set(_WIN_ARCH "${OS_ARCH}")
   if("${_WIN_ARCH}" STREQUAL "")
       # Fallback: CMAKE_HOST_SYSTEM_PROCESSOR is "AMD64" on 64-bit Windows
       if(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "AMD64")
           set(_WIN_ARCH "amd64")
       elseif(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "ARM64")
           set(_WIN_ARCH "arm64")
       endif()
   endif()

   if (MINGW AND "${_WIN_ARCH}" STREQUAL "arm64")
       set(Rust_CARGO_TARGET "aarch64-pc-windows-gnu")
   elseif (MINGW AND "${_WIN_ARCH}" STREQUAL "amd64")
       set(Rust_CARGO_TARGET "x86_64-pc-windows-gnu")
   elseif (MSVC AND "${_WIN_ARCH}" STREQUAL "arm64")
       set(Rust_CARGO_TARGET "aarch64-pc-windows-msvc")
   elseif (MSVC AND "${_WIN_ARCH}" STREQUAL "amd64")
       set(Rust_CARGO_TARGET "x86_64-pc-windows-msvc")
   endif()
endif()

message(STATUS "Rust_CARGO_TARGET: ${Rust_CARGO_TARGET}")
message(STATUS "OS_NAME: ${OS_NAME}")
message(STATUS "OS_ARCH: ${OS_ARCH}")
message(STATUS "OSX_BUILD_ARCH: ${OSX_BUILD_ARCH}")

# Use Corrosion to build Rust library
include(FetchContent)

FetchContent_Declare(
    Corrosion
    GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
    GIT_TAG v0.5.2
)
FetchContent_MakeAvailable(Corrosion)

# Import the Rust crate
corrosion_import_crate(MANIFEST_PATH "${CMAKE_CURRENT_SOURCE_DIR}/yardstick-rs/Cargo.toml"
    CRATES "yardstick"
)

# Avoid MSVC import-lib name clashes with the loadable extension output.
if(WIN32 AND TARGET yardstick-static)
    set_target_properties(yardstick-static PROPERTIES OUTPUT_NAME yardstick_rust)
endif()

# Include Rust library headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/yardstick-rs/include)

set(EXTENSION_SOURCES
    src/yardstick_extension.cpp
    src/yardstick_parser_ffi.cpp
)

if(NOT COMMAND build_static_extension)
    function(build_static_extension NAME)
        set(FILES "${ARGV}")
        list(REMOVE_AT FILES 0)
        add_library(${NAME}_extension STATIC ${FILES})
    endfunction()

    function(build_loadable_extension NAME PARAMETERS)
        set(FILES "${ARGV}")
        list(REMOVE_AT FILES 0 1)
        set(TARGET_NAME ${NAME}_loadable_extension)
        add_library(${TARGET_NAME} SHARED ${FILES})
        set_target_properties(${TARGET_NAME} PROPERTIES DEFINE_SYMBOL "")
        set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME ${NAME})
        set_target_properties(${TARGET_NAME} PROPERTIES PREFIX "")
        set_target_properties(${TARGET_NAME} PROPERTIES SUFFIX ".duckdb_extension")
        target_compile_definitions(${TARGET_NAME} PUBLIC -DDUCKDB_BUILD_LOADABLE_EXTENSION)

        if(APPLE)
            set_target_properties(${TARGET_NAME} PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
        elseif(MSVC)
            target_link_options(${TARGET_NAME} PRIVATE "/FORCE:UNRESOLVED")
            set_target_properties(
                ${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG
                "${CMAKE_BINARY_DIR}")
            set_target_properties(
                ${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE
                "${CMAKE_BINARY_DIR}")
            set_target_properties(
                ${TARGET_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY_DEBUG
                "${CMAKE_BINARY_DIR}")
            set_target_properties(
                ${TARGET_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY_RELEASE
                "${CMAKE_BINARY_DIR}")
        endif()
    endfunction()
endif()

build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

# Link the Rust static library
target_link_libraries(${EXTENSION_NAME} yardstick-static)
target_link_libraries(${LOADABLE_EXTENSION_NAME} yardstick)

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

# On Windows, we need these system libraries for Rust
if(WIN32)
    target_link_libraries(${EXTENSION_NAME} ws2_32 userenv bcrypt ntdll)
    target_link_libraries(${LOADABLE_EXTENSION_NAME} ws2_32 userenv bcrypt ntdll)
    # Avoid import lib name collision with Rust staticlib (yardstick.lib).
    set_target_properties(${LOADABLE_EXTENSION_NAME} PROPERTIES
        ARCHIVE_OUTPUT_NAME "${TARGET_NAME}_loadable")
endif()

install(
  TARGETS ${EXTENSION_NAME}
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
