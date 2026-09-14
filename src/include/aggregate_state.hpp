#pragma once

#include "yardstick_compat.hpp"

namespace duckdb {
class ExtensionLoader;

// Composite state helpers depend on the typed aggregate states in the native frontend target.
void RegisterYardstickAggregateStateFunctions(ExtensionLoader &loader);
} // namespace duckdb
