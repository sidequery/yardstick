#pragma once

#include "frontend_peg.hpp"

namespace duckdb {
#if YARDSTICK_GRAMMAR_EXTENSION
string DecorateYardstickMeasureExpression(const string &measure_expression, const string &call_sql,
                                         const vector<pair<string, string>> &dimensions,
                                         const vector<string> &local_qualifiers, const ParserOptions &options,
                                         bool preserve_dimension_qualifiers = false);
#endif
} // namespace duckdb
