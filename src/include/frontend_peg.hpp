#pragma once

#include "yardstick_compat.hpp"
#include "duckdb/parser/parser_extension.hpp"

namespace duckdb {
class DatabaseInstance;

#if YARDSTICK_GRAMMAR_EXTENSION
shared_ptr<ParserExtensionInfo> RegisterYardstickGrammar(DatabaseInstance &db);

// Recognize custom syntax with DuckDB's grammar, then adapt its source spans to
// the existing semantic lowerer. False retains the legacy frontend.
bool NormalizeYardstickGrammar(ParserExtensionInfo *info, const ParserOptions &options,
                              string &sql, bool &has_measure);
#endif
} // namespace duckdb
