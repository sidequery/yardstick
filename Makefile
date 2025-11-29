PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Configuration of extension
EXT_NAME=yardstick
EXT_CONFIG=${PROJ_DIR}extension_config.cmake

# Rust library
RUST_DIR := $(PROJ_DIR)yardstick-rs
RUST_LIB := $(RUST_DIR)/target/release/libyardstick.a

# Build Rust library first
.PHONY: rust
rust:
	cd $(RUST_DIR) && cargo build --release

# Make the DuckDB extension depend on Rust
release: rust
debug: rust

# Include the Makefile from extension-ci-tools
include extension-ci-tools/makefiles/duckdb_extension.Makefile

# Test target
.PHONY: test
test: release
	./build/release/test/unittest

# Clean everything including Rust
.PHONY: clean-all
clean-all: clean
	cd $(RUST_DIR) && cargo clean
