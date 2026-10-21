#pragma once

#include "yardstick_compat.hpp"
#include "duckdb/parser/parser.hpp"

namespace duckdb {

inline ParserOptions YardstickBuiltinParserOptions() {
#if YARDSTICK_PARSER_INSTANCE_API
    return ParserOptions::Builtin();
#else
    return ParserOptions();
#endif
}

inline vector<unique_ptr<ParsedExpression>> YardstickParseExpressionList(
    const string &sql, const ParserOptions &options = YardstickBuiltinParserOptions()) {
#if YARDSTICK_PARSER_INSTANCE_API
    Parser parser(options);
    return parser.ParseExpressionList(sql);
#else
    return Parser::ParseExpressionList(sql, options);
#endif
}

} // namespace duckdb
