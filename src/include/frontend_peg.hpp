#pragma once

#include "yardstick_compat.hpp"
#include "yardstick_ffi.h"
#include "duckdb/parser/parser_extension.hpp"
#include "duckdb/parser/parser_options.hpp"

namespace duckdb {
class DatabaseInstance;

#if YARDSTICK_GRAMMAR_EXTENSION
shared_ptr<ParserExtensionInfo> RegisterYardstickGrammar(DatabaseInstance &db);

// Keep native parsing scoped to the override's database, active grammar, and
// parser settings. Nested overrides restore the previous scope on return.
class NativeYardstickParseScope {
public:
    NativeYardstickParseScope(ParserExtensionInfo *info, const ParserOptions &options);
    ~NativeYardstickParseScope();
    NativeYardstickParseScope(const NativeYardstickParseScope &) = delete;
    NativeYardstickParseScope &operator=(const NativeYardstickParseScope &) = delete;

    const ParserOptions &ParserConfig() const { return options; }

private:
    friend YardstickAggregateCallList *FindNativeYardstickAggregates(const char *sql);
    ParserOptions options;
    const NativeYardstickParseScope *previous;
    bool available;
};

const ParserOptions *CurrentNativeYardstickParserOptions();

// Returns a complete native result, freed with yardstick_free_aggregate_list,
// or nullptr when no native scope is available or the syntax is unsupported.
YardstickAggregateCallList *FindNativeYardstickAggregates(const char *sql);

// Recognize custom syntax with DuckDB's grammar, then adapt its source spans to
// the existing semantic lowerer. False retains the legacy frontend.
bool NormalizeYardstickGrammar(ParserExtensionInfo *info, const ParserOptions &options,
                              string &sql, bool &has_measure);
#endif
} // namespace duckdb
