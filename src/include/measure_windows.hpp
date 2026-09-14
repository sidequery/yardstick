#pragma once

#include "yardstick_compat.hpp"

#if YARDSTICK_GRAMMAR_EXTENSION
#include "duckdb/common/common.hpp"
#include "duckdb/common/pair.hpp"
#include "duckdb/parser/parser_options.hpp"

namespace duckdb {

enum class WindowContextType { ALL, ALL_GLOBAL, SET, WHERE, VISIBLE };

struct WindowContextModifier {
    WindowContextType type;
    string dimension;
    string value;
};

struct MeasureWindowSource {
    string key;
    string relation_name;
    string alias;
    string clean_select_sql;
    bool grouped = false;
    vector<pair<string, string>> dimensions;
};

struct MeasureWindowCall {
    string marker_name;
    string source_key;
    string expression_sql;
    vector<WindowContextModifier> modifiers;
};

// Windows select a set of original base-row identities. Repeated identities
// introduced by joins never multiply a measure, while identical original rows
// retain their independent identities.
string RewriteNativeMeasureWindows(const string &scope_sql, const vector<MeasureWindowSource> &sources,
                                   const vector<MeasureWindowCall> &calls, const ParserOptions &options);

} // namespace duckdb
#endif
