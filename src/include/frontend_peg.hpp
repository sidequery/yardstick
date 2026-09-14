#pragma once

#include "yardstick_compat.hpp"
#include "yardstick_ffi.h"
#include "duckdb/parser/parser_extension.hpp"
#include "duckdb/parser/parser_options.hpp"

namespace duckdb {
class DatabaseInstance;
bool IsYardstickStandardAggregate(const string &name);
class Parser;

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
    friend YardstickCreateViewInfo *FindNativeYardstickMeasures(const char *sql);
    friend bool ParseNativeYardstickQuery(const string &sql, Parser &parser);
    friend YardstickCurrentReferenceList *FindNativeYardstickCurrentReferences(const char *expression);
    friend YardstickQueryScopeList *FindNativeYardstickQueryScopes(const char *sql);
    ParserOptions options;
    const NativeYardstickParseScope *previous;
    bool available;
};

const ParserOptions *CurrentNativeYardstickParserOptions();

// Binding-only schema inspection must use the originating session/transaction.
class NativeYardstickBindScope {
public:
    explicit NativeYardstickBindScope(ClientContext &context);
    ~NativeYardstickBindScope();
private:
    ClientContext *previous;
};
ClientContext *CurrentNativeYardstickClientContext();

// Parse through the active grammar while retaining Yardstick syntax capture.
bool ParseNativeYardstickQuery(const string &sql, Parser &parser);

// Returns a complete native result or recognized semantic error, freed with
// yardstick_free_aggregate_list. nullptr retains compatibility for unavailable
// native grammar or source forms outside the native adapter.
YardstickAggregateCallList *FindNativeYardstickAggregates(const char *sql);

// Source-preserving measure declarations. nullptr retains the legacy parser.
YardstickCreateViewInfo *FindNativeYardstickMeasures(const char *sql);
YardstickCurrentReferenceList *FindNativeYardstickCurrentReferences(const char *expression);
YardstickQueryScopeList *FindNativeYardstickQueryScopes(const char *sql);

// Recognize custom syntax with DuckDB's grammar, then adapt its source spans to
// the existing semantic lowerer. False retains the legacy frontend.
bool NormalizeYardstickGrammar(ParserExtensionInfo *info, const ParserOptions &options,
                              string &sql, bool &has_measure);
#endif
} // namespace duckdb
