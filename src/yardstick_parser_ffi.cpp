/**
 * yardstick_parser_ffi.cpp - C FFI implementation using DuckDB's native parser
 *
 * This file provides FFI functions that use DuckDB's parser to analyze SQL
 * and return results as C structs that can be consumed by Rust.
 *
 * Flow: Rust code -> calls C FFI functions -> C++ uses duckdb::Parser -> returns C structs to Rust
 */

#include "yardstick_ffi.h"
#include "yardstick_compat.hpp"
#include "frontend_peg.hpp"
#if YARDSTICK_GRAMMAR_EXTENSION
#include "duckdb/parser/peg/compiled_grammar.hpp"
#include "duckdb/parser/statement/insert_statement.hpp"
#include "duckdb/parser/query_node/set_operation_node.hpp"
#include "duckdb/parser/query_node/recursive_cte_node.hpp"
#include "duckdb/parser/tableref/list.hpp"
#include "duckdb/parser/tableref/table_function_ref.hpp"
#include "duckdb/parser/tableref/expressionlistref.hpp"
#include "duckdb/parser/expression/lambda_expression.hpp"
#endif

#include "duckdb/parser/parser.hpp"
#include "duckdb/parser/query_node.hpp"
#include "duckdb/parser/query_node/select_node.hpp"
#include "duckdb/parser/statement/select_statement.hpp"
#include "duckdb/parser/parsed_expression.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/parser/expression/columnref_expression.hpp"
#include "duckdb/parser/expression/star_expression.hpp"
#include "duckdb/parser/expression/comparison_expression.hpp"
#include "duckdb/parser/expression/conjunction_expression.hpp"
#include "duckdb/parser/expression/operator_expression.hpp"
#include "duckdb/parser/expression/case_expression.hpp"
#include "duckdb/parser/expression/cast_expression.hpp"
#include "duckdb/parser/expression/subquery_expression.hpp"
#include "duckdb/parser/expression/window_expression.hpp"
#include "duckdb/parser/expression/between_expression.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/parser/parsed_expression_iterator.hpp"
#include "duckdb/parser/tableref.hpp"
#include "duckdb/parser/tableref/basetableref.hpp"
#include "duckdb/parser/tableref/joinref.hpp"
#include "duckdb/parser/tableref/subqueryref.hpp"
#include "duckdb/parser/group_by_node.hpp"
#include "duckdb/common/string_util.hpp"

#include <cctype>
#include <algorithm>
#include <cstring>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>

using namespace duckdb;

//=============================================================================
// Compatibility shims for DuckDB's expression-API refactor.
//
// DuckDB main made the ParsedExpression subclass fields private (exposing
// accessors instead) and introduced a dedicated Identifier type in place of
// std::string for names. DuckDB 1.5 and earlier expose public fields and use
// std::string. CMake probes the expression accessors directly and we route
// field access through these helpers, so the extension builds against both the
// stable line and main. Remove this shim once the minimum supported DuckDB has
// the new API.
//=============================================================================

namespace {

// Rewrites must preserve the caller's grammar while avoiding parser-override
// recursion. Outside an override, legacy callers retain the default parser.
static ParserOptions YardstickParserOptions() {
#if YARDSTICK_GRAMMAR_EXTENSION
    if (auto *options = CurrentNativeYardstickParserOptions()) {
        return *options;
    }
#endif
    return ParserOptions();
}

// Identifier/string -> raw string (case preserved). The Identifier overload only
// exists on the new API; the std::string overload covers the old API and any
// already-raw string.
#if YARDSTICK_NEW_EXPR_API
inline const std::string &YsName(const Identifier &id) { return id.GetIdentifierName(); }
#endif
inline const std::string &YsName(const std::string &s) { return s; }

inline const std::string &YsFuncName(const FunctionExpression &f) {
#if YARDSTICK_NEW_EXPR_API
    return f.FunctionName().GetIdentifierName();
#else
    return f.function_name;
#endif
}

// Direct child expressions (arguments) of a function/window call.
inline std::vector<ParsedExpression *> YsArgs(FunctionExpression &f) {
    std::vector<ParsedExpression *> out;
#if YARDSTICK_NEW_EXPR_API
    for (auto &arg : f.GetArgumentsMutable()) out.push_back(arg.GetExpressionMutable().get());
#else
    for (auto &child : f.children) out.push_back(child.get());
#endif
    return out;
}
inline std::vector<ParsedExpression *> YsArgs(WindowExpression &w) {
    std::vector<ParsedExpression *> out;
#if YARDSTICK_NEW_EXPR_API
    for (auto &arg : w.GetArgumentsMutable()) out.push_back(arg.GetExpressionMutable().get());
#else
    for (auto &child : w.children) out.push_back(child.get());
#endif
    return out;
}

inline ParsedExpression *YsFilter(FunctionExpression &f) {
#if YARDSTICK_NEW_EXPR_API
    return f.FilterMutable().get();
#else
    return f.filter.get();
#endif
}
inline ParsedExpression *YsFilter(WindowExpression &w) {
#if YARDSTICK_NEW_EXPR_API
    return w.FilterMutable().get();
#else
    return w.filter_expr.get();
#endif
}

inline ParsedExpression *YsLeft(ComparisonExpression &c) {
#if YARDSTICK_NEW_EXPR_API
    return c.LeftMutable().get();
#else
    return c.left.get();
#endif
}
inline ParsedExpression *YsRight(ComparisonExpression &c) {
#if YARDSTICK_NEW_EXPR_API
    return c.RightMutable().get();
#else
    return c.right.get();
#endif
}

inline std::vector<ParsedExpression *> YsChildren(ConjunctionExpression &c) {
    std::vector<ParsedExpression *> out;
#if YARDSTICK_NEW_EXPR_API
    for (auto &child : c.GetChildrenMutable()) out.push_back(child.get());
#else
    for (auto &child : c.children) out.push_back(child.get());
#endif
    return out;
}
inline std::vector<ParsedExpression *> YsChildren(OperatorExpression &o) {
    std::vector<ParsedExpression *> out;
#if YARDSTICK_NEW_EXPR_API
    for (auto &child : o.GetChildrenMutable()) out.push_back(child.get());
#else
    for (auto &child : o.children) out.push_back(child.get());
#endif
    return out;
}
inline std::vector<ParsedExpression *> YsPartitions(WindowExpression &w) {
    std::vector<ParsedExpression *> out;
#if YARDSTICK_NEW_EXPR_API
    for (auto &part : w.PartitionsMutable()) out.push_back(part.get());
#else
    for (auto &part : w.partitions) out.push_back(part.get());
#endif
    return out;
}

inline std::vector<CaseCheck> &YsCaseChecks(CaseExpression &c) {
#if YARDSTICK_NEW_EXPR_API
    return c.CaseChecksMutable();
#else
    return c.case_checks;
#endif
}
inline ParsedExpression *YsElse(CaseExpression &c) {
#if YARDSTICK_NEW_EXPR_API
    return c.ElseMutable().get();
#else
    return c.else_expr.get();
#endif
}

inline ParsedExpression *YsChild(CastExpression &c) {
#if YARDSTICK_NEW_EXPR_API
    return c.ChildMutable().get();
#else
    return c.child.get();
#endif
}
inline ParsedExpression *YsChild(SubqueryExpression &s) {
#if YARDSTICK_NEW_EXPR_API
    return s.GetChildMutable().get();
#else
    return s.child.get();
#endif
}
inline const ParsedExpression *YsChild(const SubqueryExpression &s) {
#if YARDSTICK_NEW_EXPR_API
    return s.GetChild().get();
#else
    return s.child.get();
#endif
}
// Mutable child slot of a subquery, for in-place rewrites.
inline unique_ptr<ParsedExpression> &YsChildRef(SubqueryExpression &s) {
#if YARDSTICK_NEW_EXPR_API
    return s.GetChildMutable();
#else
    return s.child;
#endif
}

inline ParsedExpression *YsInput(BetweenExpression &b) {
#if YARDSTICK_NEW_EXPR_API
    return b.InputMutable().get();
#else
    return b.input.get();
#endif
}
inline ParsedExpression *YsLower(BetweenExpression &b) {
#if YARDSTICK_NEW_EXPR_API
    return b.LowerBoundMutable().get();
#else
    return b.lower.get();
#endif
}
inline ParsedExpression *YsUpper(BetweenExpression &b) {
#if YARDSTICK_NEW_EXPR_API
    return b.UpperBoundMutable().get();
#else
    return b.upper.get();
#endif
}

inline size_t YsColumnNameCount(const ColumnRefExpression &c) {
#if YARDSTICK_NEW_EXPR_API
    return c.ColumnNames().size();
#else
    return c.column_names.size();
#endif
}
inline const std::string &YsTableQualifier(const ColumnRefExpression &c) {
#if YARDSTICK_NEW_EXPR_API
    return YsName(c.ColumnNames()[c.ColumnNames().size() - 2]);
#else
    return c.GetTableName();
#endif
}
inline void YsPrependQualifier(ColumnRefExpression &c, const std::string &qualifier) {
#if YARDSTICK_NEW_EXPR_API
    c.ColumnNamesMutable().insert(c.ColumnNamesMutable().begin(), Identifier(qualifier));
#else
    c.column_names.insert(c.column_names.begin(), qualifier);
#endif
}

inline const std::string &YsBaseTableName(const BaseTableRef &b) {
#if YARDSTICK_NEW_EXPR_API
    return b.Table().GetIdentifierName();
#else
    return b.table_name;
#endif
}

} // namespace

//=============================================================================
// Helper: Safe strdup that handles nullptr
//=============================================================================

static char* safe_strdup(const char* s) {
    if (!s) return nullptr;
    return strdup(s);
}

static char* safe_strdup(const std::string& s) {
    return strdup(s.c_str());
}

static bool IsBoundaryChar(char c) {
    return !std::isalnum(static_cast<unsigned char>(c)) && c != '_';
}

static std::string NormalizeAliasName(const std::string &alias) {
    return StringUtil::Lower(alias);
}

using TableQualifierSet = std::unordered_set<std::string>;

static bool IsPotentialOrderAliasRef(
    const ColumnRefExpression &colref,
    const TableQualifierSet &from_qualifiers,
    std::string &alias_name
) {
    if (!colref.IsQualified()) {
        alias_name = YsName(colref.GetColumnName());
        return true;
    }
    if (YsColumnNameCount(colref) == 2 &&
        StringUtil::CIEquals(YsTableQualifier(colref), "alias")) {
        if (from_qualifiers.find(NormalizeAliasName(YsTableQualifier(colref))) !=
            from_qualifiers.end()) {
            return false;
        }
        alias_name = YsName(colref.GetColumnName());
        return true;
    }
    return false;
}

static size_t SkipWhitespaceAndComments(const std::string& sql, size_t idx) {
    while (idx < sql.size()) {
        if (std::isspace(static_cast<unsigned char>(sql[idx]))) {
            idx++;
            continue;
        }
        if (sql[idx] == '-' && idx + 1 < sql.size() && sql[idx + 1] == '-') {
            idx += 2;
            while (idx < sql.size() && sql[idx] != '\n' && sql[idx] != '\r') {
                idx++;
            }
            continue;
        }
        if (sql[idx] == '/' && idx + 1 < sql.size() && sql[idx + 1] == '*') {
            idx += 2;
            while (idx + 1 < sql.size() && !(sql[idx] == '*' && sql[idx + 1] == '/')) {
                idx++;
            }
            if (idx + 1 < sql.size()) {
                idx += 2;
            } else {
                idx = sql.size();
            }
            continue;
        }
        break;
    }
    return idx;
}

static size_t FindMatchingParen(const std::string& sql, size_t open_pos) {
    size_t depth = 0;
    bool in_single = false;
    bool in_double = false;
    bool in_backtick = false;
    bool in_bracket = false;
    bool in_line_comment = false;
    bool in_block_comment = false;

    for (size_t i = open_pos; i < sql.size(); i++) {
        char c = sql[i];

        if (in_line_comment) {
            if (c == '\n' || c == '\r') {
                in_line_comment = false;
            }
            continue;
        }
        if (in_block_comment) {
            if (c == '*' && i + 1 < sql.size() && sql[i + 1] == '/') {
                in_block_comment = false;
                i++;
            }
            continue;
        }

        if (in_single) {
            if (c == '\'') {
                if (i + 1 < sql.size() && sql[i + 1] == '\'') {
                    i++;
                } else {
                    in_single = false;
                }
            }
            continue;
        }
        if (in_double) {
            if (c == '"') {
                if (i + 1 < sql.size() && sql[i + 1] == '"') {
                    i++;
                } else {
                    in_double = false;
                }
            }
            continue;
        }
        if (in_backtick) {
            if (c == '`') {
                in_backtick = false;
            }
            continue;
        }
        if (in_bracket) {
            if (c == ']') {
                in_bracket = false;
            }
            continue;
        }

        if (c == '\'') {
            in_single = true;
            continue;
        }
        if (c == '"') {
            in_double = true;
            continue;
        }
        if (c == '`') {
            in_backtick = true;
            continue;
        }
        if (c == '[') {
            in_bracket = true;
            continue;
        }
        if (c == '-' && i + 1 < sql.size() && sql[i + 1] == '-') {
            in_line_comment = true;
            i++;
            continue;
        }
        if (c == '/' && i + 1 < sql.size() && sql[i + 1] == '*') {
            in_block_comment = true;
            i++;
            continue;
        }

        if (c == '(') {
            depth++;
            continue;
        }
        if (c == ')') {
            if (depth == 0) {
                return i;
            }
            depth--;
            if (depth == 0) {
                return i;
            }
        }
    }

    return std::string::npos;
}

static size_t FindTopLevelFrom(const std::string& sql) {
    std::string upper = StringUtil::Upper(sql);
    const std::string keyword = "FROM";

    size_t depth = 0;
    bool in_single = false;
    bool in_double = false;
    bool in_backtick = false;
    bool in_bracket = false;
    bool in_line_comment = false;
    bool in_block_comment = false;

    for (size_t i = 0; i + keyword.size() <= upper.size(); i++) {
        char c = sql[i];

        if (in_line_comment) {
            if (c == '\n' || c == '\r') {
                in_line_comment = false;
            }
            continue;
        }
        if (in_block_comment) {
            if (c == '*' && i + 1 < sql.size() && sql[i + 1] == '/') {
                in_block_comment = false;
                i++;
            }
            continue;
        }

        if (in_single) {
            if (c == '\'') {
                if (i + 1 < sql.size() && sql[i + 1] == '\'') {
                    i++;
                } else {
                    in_single = false;
                }
            }
            continue;
        }
        if (in_double) {
            if (c == '"') {
                if (i + 1 < upper.size() && upper[i + 1] == '"') {
                    i++;
                } else {
                    in_double = false;
                }
            }
            continue;
        }
        if (in_backtick) {
            if (c == '`') {
                in_backtick = false;
            }
            continue;
        }
        if (in_bracket) {
            if (c == ']') {
                in_bracket = false;
            }
            continue;
        }

        if (c == '\'') {
            in_single = true;
            continue;
        }
        if (c == '"') {
            in_double = true;
            continue;
        }
        if (c == '-' && i + 1 < sql.size() && sql[i + 1] == '-') {
            in_line_comment = true;
            i++;
            continue;
        }
        if (c == '/' && i + 1 < sql.size() && sql[i + 1] == '*') {
            in_block_comment = true;
            i++;
            continue;
        }
        if (c == '`') {
            in_backtick = true;
            continue;
        }
        if (c == '[') {
            in_bracket = true;
            continue;
        }

        if (c == '(') {
            depth++;
            continue;
        }
        if (c == ')') {
            if (depth > 0) {
                depth--;
            }
            continue;
        }

        if (depth == 0 && upper.compare(i, keyword.size(), keyword) == 0) {
            char prev = i == 0 ? '\0' : upper[i - 1];
            char next = i + keyword.size() < upper.size() ? upper[i + keyword.size()] : '\0';
            if (IsBoundaryChar(prev) && IsBoundaryChar(next)) {
                return i;
            }
        }
    }

    return std::string::npos;
}

static size_t FindSelectItemEnd(const std::string& sql, size_t start, size_t from_pos) {
    if (start >= sql.size()) {
        return start;
    }
    size_t limit = from_pos == std::string::npos ? sql.size() : from_pos;
    size_t depth = 0;
    bool in_single = false;
    bool in_double = false;
    bool in_backtick = false;
    bool in_bracket = false;
    bool in_line_comment = false;
    bool in_block_comment = false;

    for (size_t i = start; i < limit; i++) {
        char c = sql[i];

        if (in_line_comment) {
            if (c == '\n' || c == '\r') {
                in_line_comment = false;
            }
            continue;
        }
        if (in_block_comment) {
            if (c == '*' && i + 1 < limit && sql[i + 1] == '/') {
                in_block_comment = false;
                i++;
            }
            continue;
        }

        if (in_single) {
            if (c == '\'') {
                if (i + 1 < limit && sql[i + 1] == '\'') {
                    i++;
                } else {
                    in_single = false;
                }
            }
            continue;
        }
        if (in_double) {
            if (c == '"') {
                if (i + 1 < limit && sql[i + 1] == '"') {
                    i++;
                } else {
                    in_double = false;
                }
            }
            continue;
        }
        if (in_backtick) {
            if (c == '`') {
                in_backtick = false;
            }
            continue;
        }
        if (in_bracket) {
            if (c == ']') {
                in_bracket = false;
            }
            continue;
        }

        if (c == '\'') {
            in_single = true;
            continue;
        }
        if (c == '"') {
            in_double = true;
            continue;
        }
        if (c == '-' && i + 1 < limit && sql[i + 1] == '-') {
            in_line_comment = true;
            i++;
            continue;
        }
        if (c == '/' && i + 1 < limit && sql[i + 1] == '*') {
            in_block_comment = true;
            i++;
            continue;
        }
        if (c == '`') {
            in_backtick = true;
            continue;
        }
        if (c == '[') {
            in_bracket = true;
            continue;
        }

        if (c == '(') {
            depth++;
            continue;
        }
        if (c == ')') {
            if (depth > 0) {
                depth--;
            }
            continue;
        }

        if (depth == 0 && c == ',') {
            return i;
        }
    }

    return limit;
}

static size_t FindAggregateCallEnd(const std::string& sql, size_t start) {
    std::string upper = StringUtil::Upper(sql);
    const std::string keyword = "AGGREGATE";
    if (start >= sql.size() || upper.compare(start, keyword.size(), keyword) != 0) {
        return start;
    }

    size_t i = SkipWhitespaceAndComments(sql, start + keyword.size());
    if (i >= sql.size() || sql[i] != '(') {
        return start;
    }

    size_t close = FindMatchingParen(sql, i);
    if (close == std::string::npos) {
        return start;
    }
    size_t end = close + 1;

    while (end < sql.size()) {
        size_t j = SkipWhitespaceAndComments(sql, end);
        if (j + 2 > sql.size()) {
            break;
        }
        if (upper.compare(j, 2, "AT") != 0) {
            break;
        }
        size_t k = SkipWhitespaceAndComments(sql, j + 2);
        if (k >= sql.size() || sql[k] != '(') {
            break;
        }
        size_t at_close = FindMatchingParen(sql, k);
        if (at_close == std::string::npos) {
            break;
        }
        end = at_close + 1;
    }

    return end;
}

//=============================================================================
// Helper: Check if expression is a standard aggregate function
//=============================================================================

static bool IsStandardAggregate(const std::string& name) {
    static const std::vector<std::string> aggregates = {
        "sum", "count", "avg", "min", "max",
        "first", "last", "any_value",
        "stddev", "stddev_pop", "stddev_samp",
        "variance", "var_pop", "var_samp",
        "string_agg", "listagg", "group_concat",
        "array_agg", "list", "count_star", "median", "mode",
        "quantile", "quantile_cont", "quantile_disc", "percentile_cont", "percentile_disc",
        "bool_and", "bool_or", "bit_and", "bit_or", "bit_xor"
    };
    std::string lower = StringUtil::Lower(name);
    for (const auto& agg : aggregates) {
        if (lower == agg) return true;
    }
    return false;
}

//=============================================================================
// Forward declarations for recursive AST walking
//=============================================================================

struct AggregateCallInfo {
    std::string measure_name;
    uint32_t start_pos;
    uint32_t end_pos;
    std::vector<YardstickAtModifier> modifiers;
};

static void FindAggregateCalls(ParsedExpression* expr, std::vector<AggregateCallInfo>& results,
                               const std::string& sql);
static void CollectTablesFromTableRef(TableRef* ref, std::vector<YardstickTableRef>& tables,
                                     const std::unordered_set<std::string>* cte_names = nullptr);
static bool ExpressionContainsAggregate(ParsedExpression* expr);
static bool ExpressionContainsMeasureRef(ParsedExpression* expr);
static void QualifyColumnRefs(ParsedExpression* expr, const std::string& qualifier);

//=============================================================================
// AST Walking: Find AGGREGATE() function calls
//=============================================================================

static void FindAggregateCalls(ParsedExpression* expr, std::vector<AggregateCallInfo>& results,
                               const std::string& sql) {
    if (!expr) return;

    switch (expr->GetExpressionClass()) {
        case ExpressionClass::FUNCTION: {
            auto* func = static_cast<FunctionExpression*>(expr);
            std::string lower_name = StringUtil::Lower(YsFuncName(*func));
            auto args = YsArgs(*func);

            if (lower_name == "aggregate") {
                AggregateCallInfo info;

                // Get measure name from first argument
                if (!args.empty()) {
                    // First argument should be measure name (column ref or string)
                    auto* first_arg = args[0];
                    if (first_arg->GetExpressionClass() == ExpressionClass::COLUMN_REF) {
                        auto* col = static_cast<ColumnRefExpression*>(first_arg);
                        info.measure_name = YsName(col->GetColumnName());
                    } else {
                        info.measure_name = first_arg->ToString();
                    }
                }

                // Get position from query_location
                optional_idx query_location = expr->GetQueryLocation();
                if (query_location.IsValid()) {
                    info.start_pos = static_cast<uint32_t>(query_location.GetIndex());
                } else {
                    info.start_pos = 0;
                }
                size_t end_pos = FindAggregateCallEnd(sql, info.start_pos);
                info.end_pos = static_cast<uint32_t>(end_pos);

                // Parse AT modifiers from remaining arguments
                // AT syntax in DuckDB typically appears as special function arguments
                // For now, we look for patterns like AT(ALL), AT(dimension), etc.
                for (size_t i = 1; i < args.size(); i++) {
                    auto* arg = args[i];

                    // Check if this is an AT modifier call
                    if (arg->GetExpressionClass() == ExpressionClass::FUNCTION) {
                        auto* at_func = static_cast<FunctionExpression*>(arg);
                        std::string at_name = StringUtil::Lower(YsFuncName(*at_func));

                        if (at_name == "at") {
                            YardstickAtModifier mod;
                            mod.dimension = nullptr;
                            mod.value = nullptr;

                            auto at_args = YsArgs(*at_func);
                            if (!at_args.empty()) {
                                auto* at_arg = at_args[0];
                                std::string at_arg_str = at_arg->ToString();
                                std::string at_arg_lower = StringUtil::Lower(at_arg_str);

                                if (at_arg_lower == "all" || at_arg_lower == "(all)") {
                                    mod.type = YARDSTICK_AT_ALL_GLOBAL;
                                } else if (at_arg_lower == "visible" || at_arg_lower == "(visible)") {
                                    mod.type = YARDSTICK_AT_VISIBLE;
                                } else if (at_arg_str.find("=") != std::string::npos) {
                                    // SET modifier: dimension = value
                                    mod.type = YARDSTICK_AT_SET;
                                    size_t eq_pos = at_arg_str.find("=");
                                    std::string dim = at_arg_str.substr(0, eq_pos);
                                    std::string val = at_arg_str.substr(eq_pos + 1);
                                    // Trim whitespace
                                    dim.erase(0, dim.find_first_not_of(" \t"));
                                    dim.erase(dim.find_last_not_of(" \t") + 1);
                                    val.erase(0, val.find_first_not_of(" \t"));
                                    val.erase(val.find_last_not_of(" \t") + 1);
                                    mod.dimension = safe_strdup(dim);
                                    mod.value = safe_strdup(val);
                                } else if (at_args.size() >= 2) {
                                    // WHERE modifier or ALL dimension
                                    auto* second = at_args[1];
                                    if (at_arg_lower == "all") {
                                        mod.type = YARDSTICK_AT_ALL_DIM;
                                        mod.dimension = safe_strdup(second->ToString());
                                    } else if (at_arg_lower == "where") {
                                        mod.type = YARDSTICK_AT_WHERE;
                                        mod.value = safe_strdup(second->ToString());
                                    }
                                } else {
                                    // Assume it's ALL dimension with single arg
                                    mod.type = YARDSTICK_AT_ALL_DIM;
                                    mod.dimension = safe_strdup(at_arg_str);
                                }
                            } else {
                                mod.type = YARDSTICK_AT_NONE;
                            }

                            info.modifiers.push_back(mod);
                        }
                    }
                }

                results.push_back(std::move(info));
            }

            // Recurse into function children
            for (auto* child : args) {
                FindAggregateCalls(child, results, sql);
            }
            if (auto* filter = YsFilter(*func)) {
                FindAggregateCalls(filter, results, sql);
            }
            break;
        }

        case ExpressionClass::COMPARISON: {
            auto* comp = static_cast<ComparisonExpression*>(expr);
            FindAggregateCalls(YsLeft(*comp), results, sql);
            FindAggregateCalls(YsRight(*comp), results, sql);
            break;
        }

        case ExpressionClass::CONJUNCTION: {
            auto* conj = static_cast<ConjunctionExpression*>(expr);
            for (auto* child : YsChildren(*conj)) {
                FindAggregateCalls(child, results, sql);
            }
            break;
        }

        case ExpressionClass::OPERATOR: {
            auto* op = static_cast<OperatorExpression*>(expr);
            for (auto* child : YsChildren(*op)) {
                FindAggregateCalls(child, results, sql);
            }
            break;
        }

        case ExpressionClass::CASE: {
            auto* case_expr = static_cast<CaseExpression*>(expr);
            for (auto& check : YsCaseChecks(*case_expr)) {
                FindAggregateCalls(check.when_expr.get(), results, sql);
                FindAggregateCalls(check.then_expr.get(), results, sql);
            }
            if (auto* else_expr = YsElse(*case_expr)) {
                FindAggregateCalls(else_expr, results, sql);
            }
            break;
        }

        case ExpressionClass::CAST: {
            auto* cast = static_cast<CastExpression*>(expr);
            FindAggregateCalls(YsChild(*cast), results, sql);
            break;
        }

        case ExpressionClass::SUBQUERY: {
            auto* subq = static_cast<SubqueryExpression*>(expr);
            if (auto* child = YsChild(*subq)) {
                FindAggregateCalls(child, results, sql);
            }
            // Note: We don't recurse into the subquery itself
            break;
        }

        case ExpressionClass::WINDOW: {
            auto* window = static_cast<WindowExpression*>(expr);
            for (auto* child : YsArgs(*window)) {
                FindAggregateCalls(child, results, sql);
            }
            for (auto* part : YsPartitions(*window)) {
                FindAggregateCalls(part, results, sql);
            }
            if (auto* filter = YsFilter(*window)) {
                FindAggregateCalls(filter, results, sql);
            }
            break;
        }

        case ExpressionClass::BETWEEN: {
            auto* between = static_cast<BetweenExpression*>(expr);
            FindAggregateCalls(YsInput(*between), results, sql);
            FindAggregateCalls(YsLower(*between), results, sql);
            FindAggregateCalls(YsUpper(*between), results, sql);
            break;
        }

        default:
            // COLUMN_REF, CONSTANT, STAR, etc. have no children
            break;
    }
}

//=============================================================================
// AST Walking: Collect tables from FROM clause
//=============================================================================

static void CollectTablesFromTableRef(TableRef* ref, std::vector<YardstickTableRef>& tables,
                                     const std::unordered_set<std::string>* cte_names) {
    if (!ref) return;

    switch (ref->type) {
        case TableReferenceType::BASE_TABLE: {
            auto* base = static_cast<BaseTableRef*>(ref);
            YardstickTableRef t {};
            t.table_name = safe_strdup(YsBaseTableName(*base));
            t.alias = base->alias.empty() ? nullptr : safe_strdup(YsName(base->alias));
            t.is_subquery = false;
#if YARDSTICK_GRAMMAR_EXTENSION
            t.schema_qualified = !base->GetQualifiedName().Schema().empty();
            if (cte_names && base->GetQualifiedName().Schema().empty() &&
                cte_names->count(StringUtil::Lower(YsBaseTableName(*base)))) {
                // A local CTE shadows the catalog view with the same name.
                t.is_subquery = true;
            }
#endif
            tables.push_back(t);
            break;
        }

        case TableReferenceType::JOIN: {
            auto* join = static_cast<JoinRef*>(ref);
            CollectTablesFromTableRef(join->left.get(), tables, cte_names);
            CollectTablesFromTableRef(join->right.get(), tables, cte_names);
            break;
        }

        case TableReferenceType::SUBQUERY: {
            auto* subq = static_cast<SubqueryRef*>(ref);
            YardstickTableRef t {};
            t.table_name = subq->alias.empty() ? safe_strdup("(subquery)") : safe_strdup(YsName(subq->alias));
            t.alias = subq->alias.empty() ? nullptr : safe_strdup(YsName(subq->alias));
            t.is_subquery = true;
            tables.push_back(t);
            break;
        }

        default:
            // TABLE_FUNCTION, EXPRESSION_LIST, etc.
            break;
    }
}

//=============================================================================
// AST Walking: Check if expression contains aggregate functions
//=============================================================================

static bool ExpressionContainsAggregate(ParsedExpression* expr) {
    if (!expr) return false;

    switch (expr->GetExpressionClass()) {
        case ExpressionClass::FUNCTION: {
            auto* func = static_cast<FunctionExpression*>(expr);
            if (IsStandardAggregate(YsFuncName(*func))) {
                return true;
            }
            for (auto* child : YsArgs(*func)) {
                if (ExpressionContainsAggregate(child)) return true;
            }
            if (auto* filter = YsFilter(*func)) {
                if (ExpressionContainsAggregate(filter)) return true;
            }
            return false;
        }

        case ExpressionClass::COMPARISON: {
            auto* comp = static_cast<ComparisonExpression*>(expr);
            return ExpressionContainsAggregate(YsLeft(*comp)) ||
                   ExpressionContainsAggregate(YsRight(*comp));
        }

        case ExpressionClass::CONJUNCTION: {
            auto* conj = static_cast<ConjunctionExpression*>(expr);
            for (auto* child : YsChildren(*conj)) {
                if (ExpressionContainsAggregate(child)) return true;
            }
            return false;
        }

        case ExpressionClass::OPERATOR: {
            auto* op = static_cast<OperatorExpression*>(expr);
            for (auto* child : YsChildren(*op)) {
                if (ExpressionContainsAggregate(child)) return true;
            }
            return false;
        }

        case ExpressionClass::CASE: {
            auto* case_expr = static_cast<CaseExpression*>(expr);
            for (auto& check : YsCaseChecks(*case_expr)) {
                if (ExpressionContainsAggregate(check.when_expr.get())) return true;
                if (ExpressionContainsAggregate(check.then_expr.get())) return true;
            }
            auto* else_expr = YsElse(*case_expr);
            return else_expr && ExpressionContainsAggregate(else_expr);
        }

        case ExpressionClass::CAST: {
            auto* cast = static_cast<CastExpression*>(expr);
            return ExpressionContainsAggregate(YsChild(*cast));
        }

        case ExpressionClass::WINDOW:
            // Window functions are aggregates in a sense
            return true;

        case ExpressionClass::BETWEEN: {
            auto* between = static_cast<BetweenExpression*>(expr);
            return ExpressionContainsAggregate(YsInput(*between)) ||
                   ExpressionContainsAggregate(YsLower(*between)) ||
                   ExpressionContainsAggregate(YsUpper(*between));
        }

        default:
            return false;
    }
}

//=============================================================================
// AST Walking: Check if expression references AGGREGATE() function
//=============================================================================

static bool ExpressionContainsMeasureRef(ParsedExpression* expr) {
    if (!expr) return false;

    switch (expr->GetExpressionClass()) {
        case ExpressionClass::FUNCTION: {
            auto* func = static_cast<FunctionExpression*>(expr);
            auto name = StringUtil::Lower(YsFuncName(*func));
            if ((name == "aggregate" && YsArgs(*func).size() == 1) ||
                name == "__yardstick_at" || name == "__yardstick_measure") {
                return true;
            }
            for (auto* child : YsArgs(*func)) {
                if (ExpressionContainsMeasureRef(child)) return true;
            }
            if (auto* filter = YsFilter(*func)) {
                if (ExpressionContainsMeasureRef(filter)) return true;
            }
            return false;
        }

        case ExpressionClass::COMPARISON: {
            auto* comp = static_cast<ComparisonExpression*>(expr);
            return ExpressionContainsMeasureRef(YsLeft(*comp)) ||
                   ExpressionContainsMeasureRef(YsRight(*comp));
        }

        case ExpressionClass::CONJUNCTION: {
            auto* conj = static_cast<ConjunctionExpression*>(expr);
            for (auto* child : YsChildren(*conj)) {
                if (ExpressionContainsMeasureRef(child)) return true;
            }
            return false;
        }

        case ExpressionClass::OPERATOR: {
            auto* op = static_cast<OperatorExpression*>(expr);
            for (auto* child : YsChildren(*op)) {
                if (ExpressionContainsMeasureRef(child)) return true;
            }
            return false;
        }

        case ExpressionClass::CASE: {
            auto* case_expr = static_cast<CaseExpression*>(expr);
            for (auto& check : YsCaseChecks(*case_expr)) {
                if (ExpressionContainsMeasureRef(check.when_expr.get())) return true;
                if (ExpressionContainsMeasureRef(check.then_expr.get())) return true;
            }
            auto* else_expr = YsElse(*case_expr);
            return else_expr && ExpressionContainsMeasureRef(else_expr);
        }

        case ExpressionClass::CAST: {
            auto* cast = static_cast<CastExpression*>(expr);
            return ExpressionContainsMeasureRef(YsChild(*cast));
        }

        case ExpressionClass::BETWEEN: {
            auto* between = static_cast<BetweenExpression*>(expr);
            return ExpressionContainsMeasureRef(YsInput(*between)) ||
                   ExpressionContainsMeasureRef(YsLower(*between)) ||
                   ExpressionContainsMeasureRef(YsUpper(*between));
        }

        default:
            return false;
    }
}

static void QualifyColumnRefs(ParsedExpression* expr, const std::string& qualifier) {
    if (!expr) return;

    switch (expr->GetExpressionClass()) {
        case ExpressionClass::COLUMN_REF: {
            auto* col = static_cast<ColumnRefExpression*>(expr);
            if (YsColumnNameCount(*col) == 1) {
                YsPrependQualifier(*col, qualifier);
            }
            break;
        }
        case ExpressionClass::FUNCTION: {
            auto* func = static_cast<FunctionExpression*>(expr);
            for (auto* child : YsArgs(*func)) {
                QualifyColumnRefs(child, qualifier);
            }
            if (auto* filter = YsFilter(*func)) {
                QualifyColumnRefs(filter, qualifier);
            }
            break;
        }
        case ExpressionClass::COMPARISON: {
            auto* comp = static_cast<ComparisonExpression*>(expr);
            QualifyColumnRefs(YsLeft(*comp), qualifier);
            QualifyColumnRefs(YsRight(*comp), qualifier);
            break;
        }
        case ExpressionClass::CONJUNCTION: {
            auto* conj = static_cast<ConjunctionExpression*>(expr);
            for (auto* child : YsChildren(*conj)) {
                QualifyColumnRefs(child, qualifier);
            }
            break;
        }
        case ExpressionClass::OPERATOR: {
            auto* op = static_cast<OperatorExpression*>(expr);
            for (auto* child : YsChildren(*op)) {
                QualifyColumnRefs(child, qualifier);
            }
            break;
        }
        case ExpressionClass::CASE: {
            auto* case_expr = static_cast<CaseExpression*>(expr);
            for (auto& check : YsCaseChecks(*case_expr)) {
                QualifyColumnRefs(check.when_expr.get(), qualifier);
                QualifyColumnRefs(check.then_expr.get(), qualifier);
            }
            if (auto* else_expr = YsElse(*case_expr)) {
                QualifyColumnRefs(else_expr, qualifier);
            }
            break;
        }
        case ExpressionClass::CAST: {
            auto* cast = static_cast<CastExpression*>(expr);
            QualifyColumnRefs(YsChild(*cast), qualifier);
            break;
        }
        case ExpressionClass::SUBQUERY: {
            auto* subq = static_cast<SubqueryExpression*>(expr);
            if (auto* child = YsChild(*subq)) {
                QualifyColumnRefs(child, qualifier);
            }
            break;
        }
        case ExpressionClass::WINDOW: {
            auto* window = static_cast<WindowExpression*>(expr);
            for (auto* child : YsArgs(*window)) {
                QualifyColumnRefs(child, qualifier);
            }
            for (auto* part : YsPartitions(*window)) {
                QualifyColumnRefs(part, qualifier);
            }
            if (auto* filter = YsFilter(*window)) {
                QualifyColumnRefs(filter, qualifier);
            }
            break;
        }
        case ExpressionClass::BETWEEN: {
            auto* between = static_cast<BetweenExpression*>(expr);
            QualifyColumnRefs(YsInput(*between), qualifier);
            QualifyColumnRefs(YsLower(*between), qualifier);
            QualifyColumnRefs(YsUpper(*between), qualifier);
            break;
        }
        default:
            break;
    }
}

//=============================================================================
// FFI Implementation: yardstick_find_aggregates
//=============================================================================

#if YARDSTICK_GRAMMAR_EXTENSION
namespace {
bool CurrentConstantExpression(const ParsedExpression &expression) {
    if (expression.GetExpressionClass() == ExpressionClass::CONSTANT) {
        return true;
    }
    if (expression.GetExpressionClass() == ExpressionClass::CAST) {
        return CurrentConstantExpression(expression.Cast<CastExpression>().Child());
    }
    return false;
}

bool CurrentEqualityProof(const ParsedExpression &predicate, const ColumnRefExpression &dimension,
                          const string &qualifier) {
    if (predicate.GetExpressionType() == ExpressionType::CONJUNCTION_AND) {
        for (auto &child : predicate.Cast<ConjunctionExpression>().GetChildren()) {
            if (CurrentEqualityProof(*child, dimension, qualifier)) {
                return true;
            }
        }
        return false;
    }
    if (predicate.GetExpressionType() != ExpressionType::COMPARE_EQUAL) {
        return false;
    }
    auto &comparison = predicate.Cast<ComparisonExpression>();
    auto matches = [&](const ParsedExpression &candidate, const ParsedExpression &value) {
        if (candidate.GetExpressionClass() != ExpressionClass::COLUMN_REF || !CurrentConstantExpression(value)) {
            return false;
        }
        auto &names = candidate.Cast<ColumnRefExpression>().ColumnNames();
        auto &wanted = dimension.ColumnNames();
        if (names.empty() || wanted.empty() || names.back() != wanted.back()) {
            return false;
        }
        if (names.size() == 1) {
            return true;
        }
        if (wanted.size() > 1) {
            return names == wanted;
        }
        return qualifier.empty() || names[names.size() - 2] == Identifier(qualifier);
    };
    return matches(comparison.Left(), comparison.Right()) || matches(comparison.Right(), comparison.Left());
}
} // namespace
#endif

extern "C" int32_t yardstick_current_where_is_single_valued(const char* predicate, const char* dimension,
                                                            const char* qualifier) {
#if YARDSTICK_GRAMMAR_EXTENSION
    auto *options = CurrentNativeYardstickParserOptions();
    if (!options || !options->compiled_grammar || !options->compiled_grammar->HasGrammarChanges()) {
        return -1;
    }
    try {
        auto predicates = Parser::ParseExpressionList(predicate, *options);
        auto dimensions = Parser::ParseExpressionList(dimension, *options);
        if (predicates.size() != 1 || dimensions.size() != 1 ||
            dimensions[0]->GetExpressionClass() != ExpressionClass::COLUMN_REF) {
            return 0;
        }
        return CurrentEqualityProof(*predicates[0], dimensions[0]->Cast<ColumnRefExpression>(),
                                    qualifier ? qualifier : string()) ? 1 : 0;
    } catch (const std::exception &) {
        return 0;
    }
#else
    return -1;
#endif
}

extern "C" YardstickCurrentReferenceList* yardstick_find_current_references(const char* expression) {
#if YARDSTICK_GRAMMAR_EXTENSION
    return FindNativeYardstickCurrentReferences(expression);
#else
    return nullptr;
#endif
}

extern "C" YardstickQueryScopeList* yardstick_find_query_scopes(const char* sql) {
#if YARDSTICK_GRAMMAR_EXTENSION
    return FindNativeYardstickQueryScopes(sql);
#else
    return nullptr;
#endif
}

extern "C" void yardstick_free_query_scopes(YardstickQueryScopeList* list) {
    if (!list) {
        return;
    }
    for (size_t i = 0; i < list->count; i++) {
        for (size_t j = 0; j < list->scopes[i].visible_cte_count; j++) {
            free(const_cast<char*>(list->scopes[i].visible_ctes[j]));
        }
        delete[] list->scopes[i].visible_ctes;
    }
    delete[] list->scopes;
    delete list;
}

extern "C" void yardstick_free_current_reference_list(YardstickCurrentReferenceList* list) {
    if (!list) {
        return;
    }
    for (size_t i = 0; i < list->count; i++) {
        free(const_cast<char*>(list->references[i].dimension));
    }
    delete[] list->references;
    free(const_cast<char*>(list->error));
    delete list;
}

extern "C" YardstickAggregateCallList* yardstick_find_aggregates(const char* sql) {
#if YARDSTICK_GRAMMAR_EXTENSION
    if (auto *native = FindNativeYardstickAggregates(sql)) {
        return native;
    }
#endif
    auto* result = new YardstickAggregateCallList();
    result->native_parsed = false;
    result->calls = nullptr;
    result->count = 0;
    result->error = nullptr;

    if (!sql) {
        result->error = safe_strdup("NULL SQL input");
        return result;
    }

    try {
        Parser parser(YardstickParserOptions());
        parser.ParseQuery(sql);

        if (parser.statements.empty()) {
            return result;
        }

        std::vector<AggregateCallInfo> aggregates;

        for (auto& stmt : parser.statements) {
            if (stmt->type != StatementType::SELECT_STATEMENT) {
                continue;
            }

            auto* select_stmt = static_cast<SelectStatement*>(stmt.get());
            if (!select_stmt->node || select_stmt->node->type != QueryNodeType::SELECT_NODE) {
                continue;
            }

            auto* select_node = static_cast<SelectNode*>(select_stmt->node.get());

            // Search in SELECT list
            for (auto& expr : select_node->select_list) {
                FindAggregateCalls(expr.get(), aggregates, sql);
            }

            // Search in WHERE clause
            if (select_node->where_clause) {
                FindAggregateCalls(select_node->where_clause.get(), aggregates, sql);
            }

            // Search in HAVING clause
            if (select_node->having) {
                FindAggregateCalls(select_node->having.get(), aggregates, sql);
            }

            // Search in GROUP BY expressions
            for (auto& expr : select_node->groups.group_expressions) {
                FindAggregateCalls(expr.get(), aggregates, sql);
            }
        }

        // Convert to C structs
        if (!aggregates.empty()) {
            result->count = aggregates.size();
            result->calls = new YardstickAggregateCall[result->count];

            for (size_t i = 0; i < aggregates.size(); i++) {
                auto& info = aggregates[i];
                auto& call = result->calls[i];

                call.measure_name = safe_strdup(info.measure_name);
                call.start_pos = info.start_pos;
                call.end_pos = info.end_pos;

                if (!info.modifiers.empty()) {
                    call.modifier_count = info.modifiers.size();
                    call.modifiers = new YardstickAtModifier[call.modifier_count];
                    for (size_t j = 0; j < info.modifiers.size(); j++) {
                        call.modifiers[j] = info.modifiers[j];
                    }
                } else {
                    call.modifiers = nullptr;
                    call.modifier_count = 0;
                }
            }
        }

    } catch (const std::exception& e) {
        result->error = safe_strdup(e.what());
    }

    return result;
}

//=============================================================================
// FFI Implementation: yardstick_free_aggregate_list
//=============================================================================

extern "C" void yardstick_free_aggregate_list(YardstickAggregateCallList* list) {
    if (!list) return;

    for (size_t i = 0; i < list->count; i++) {
        auto& call = list->calls[i];
        free(const_cast<char*>(call.measure_name));
        for (size_t j = 0; j < call.modifier_count; j++) {
            free(const_cast<char*>(call.modifiers[j].dimension));
            free(const_cast<char*>(call.modifiers[j].value));
        }
        delete[] call.modifiers;
    }
    delete[] list->calls;
    free(const_cast<char*>(list->error));
    delete list;
}

//=============================================================================
// FFI Implementation: yardstick_parse_select
//=============================================================================

#if YARDSTICK_GRAMMAR_EXTENSION
// Resolve only lexical ownership here. An unqualified column below a sourced
// query needs schema binding; do not guess that it belongs to the outer query.
static void NativeRelationQualifiers(TableRef &ref, std::unordered_set<string> &qualifiers) {
    if (!ref.alias.empty()) {
        qualifiers.insert(StringUtil::Lower(YsName(ref.alias)));
    } else if (ref.type == TableReferenceType::BASE_TABLE) {
        qualifiers.insert(StringUtil::Lower(YsBaseTableName(ref.Cast<BaseTableRef>())));
    } else if (ref.type == TableReferenceType::TABLE_FUNCTION) {
        auto &function = ref.Cast<TableFunctionRef>().function;
        if (function && function->GetExpressionClass() == ExpressionClass::FUNCTION) {
            qualifiers.insert(StringUtil::Lower(YsFuncName(function->Cast<FunctionExpression>())));
        }
    }
    if (ref.type == TableReferenceType::JOIN) {
        auto &join = ref.Cast<JoinRef>();
        NativeRelationQualifiers(*join.left, qualifiers);
        NativeRelationQualifiers(*join.right, qualifiers);
    } else if (ref.type == TableReferenceType::PIVOT) {
        NativeRelationQualifiers(*ref.Cast<PivotRef>().source, qualifiers);
    }
}

static vector<string> NativeSubqueryDimensions(ParsedExpression &projection, SelectNode &outer) {
    struct Scope {
        bool unqualified_outer = true;
        bool nested = false;
        std::unordered_set<string> shadowed;
        std::unordered_set<string> aliases;
    };
    std::unordered_set<string> outer_qualifiers;
    NativeRelationQualifiers(*outer.from_table, outer_qualifiers);
    vector<string> dimensions;
    std::function<void(ParsedExpression &, const Scope &)> expression;
    std::function<void(QueryNode &, Scope)> query;
    std::function<void(TableRef &, const Scope &)> table;
    expression = [&](ParsedExpression &expr, const Scope &scope) {
        if (expr.GetExpressionClass() == ExpressionClass::COLUMN_REF) {
            auto &names = expr.Cast<ColumnRefExpression>().ColumnNames();
            bool belongs_to_outer = scope.unqualified_outer && !scope.aliases.count(StringUtil::Lower(YsName(names.back())));
            if (names.size() > 1 && scope.nested) {
                // Catalog/schema prefixes precede the relation component.
                auto qualifier = StringUtil::Lower(YsName(names[names.size() - 2]));
                belongs_to_outer = outer_qualifiers.count(qualifier) && !scope.shadowed.count(qualifier);
            }
            if (belongs_to_outer) {
                auto sql = expr.ToString();
                if (std::find(dimensions.begin(), dimensions.end(), sql) == dimensions.end()) {
                    dimensions.push_back(std::move(sql));
                }
            }
        } else if (expr.GetExpressionClass() == ExpressionClass::SUBQUERY) {
            query(*expr.Cast<SubqueryExpression>().SubqueryMutable()->node, scope);
        }
        ParsedExpressionIterator::EnumerateChildren(expr, [&](ParsedExpression &child) { expression(child, scope); });
    };
    table = [&](TableRef &ref, const Scope &scope) {
        if (ref.type == TableReferenceType::SUBQUERY) {
            query(*ref.Cast<SubqueryRef>().subquery->node, scope);
        } else if (ref.type == TableReferenceType::JOIN) {
            auto &join = ref.Cast<JoinRef>();
            table(*join.left, scope);
            table(*join.right, scope);
            if (join.condition) {
                expression(*join.condition, scope);
            }
        } else if (ref.type == TableReferenceType::PIVOT) {
            auto &pivot = ref.Cast<PivotRef>();
            table(*pivot.source, scope);
            for (auto &aggregate : pivot.aggregates) {
                expression(*aggregate, scope);
            }
        } else {
            ParsedExpressionIterator::EnumerateTableRefChildren(ref,
                [&](unique_ptr<ParsedExpression> &expr) { expression(*expr, scope); });
            if (ref.type == TableReferenceType::TABLE_FUNCTION) {
                auto &subquery = ref.Cast<TableFunctionRef>().subquery;
                if (subquery) {
                    query(*subquery->node, scope);
                }
            }
        }
    };
    query = [&](QueryNode &node, Scope scope) {
        scope.nested = true;
        for (auto &cte : node.cte_map.map) {
            if (cte.second->query_node) {
                query(*cte.second->query_node, scope);
            }
        }
        if (node.type == QueryNodeType::SELECT_NODE) {
            auto &select = node.Cast<SelectNode>();
            if (select.from_table->type != TableReferenceType::EMPTY_FROM) {
                scope.unqualified_outer = false;
                NativeRelationQualifiers(*select.from_table, scope.shadowed);
            }
            for (auto &expr : select.select_list) {
                expression(*expr, scope);
                if (expr->HasAlias()) {
                    scope.aliases.insert(StringUtil::Lower(YsName(expr->GetAlias())));
                }
            }
            for (auto &expr : select.groups.group_expressions) {
                expression(*expr, scope);
            }
            for (auto *expr : {select.where_clause.get(), select.having.get(), select.qualify.get()}) {
                if (expr) {
                    expression(*expr, scope);
                }
            }
            table(*select.from_table, scope);
        } else if (node.type == QueryNodeType::SET_OPERATION_NODE) {
            for (auto &child : node.Cast<SetOperationNode>().children) {
                query(*child, scope);
            }
        } else if (node.type == QueryNodeType::RECURSIVE_CTE_NODE) {
            auto &recursive = node.Cast<RecursiveCTENode>();
            query(*recursive.left, scope);
            query(*recursive.right, scope);
        } else {
            throw ParserException("Unsupported subquery grouping scope");
        }
        ParsedExpressionIterator::EnumerateQueryNodeModifiers(node,
            [&](unique_ptr<ParsedExpression> &expr) { expression(*expr, scope); });
    };
    expression(projection, Scope {});
    return dimensions;
}

static void CollectNativeAtReferences(ParsedExpression &expression,
                                      vector<YardstickMeasureReference> &references) {
    if (expression.GetExpressionClass() == ExpressionClass::SUBQUERY) {
        // Nested query bodies have independent relation scope. An IN/ANY left
        // operand, when present, still belongs to this SELECT.
        if (auto *child = YsChild(expression.Cast<SubqueryExpression>())) {
            CollectNativeAtReferences(*child, references);
        }
        return;
    }
    if (expression.GetExpressionClass() == ExpressionClass::FUNCTION) {
        auto &function = expression.Cast<FunctionExpression>();
        if (YsFuncName(function) == "__yardstick_at") {
            auto arguments = YsArgs(function);
            if (!arguments.empty()) {
                auto &operand = *arguments.front();
                if (operand.GetExpressionClass() == ExpressionClass::COLUMN_REF) {
                    auto &names = operand.Cast<ColumnRefExpression>().ColumnNames();
                    auto location = operand.GetQueryLocation();
                    if ((names.size() == 1 || names.size() == 2) && location.IsValid()) {
                        YardstickMeasureReference reference {};
                        reference.column = safe_strdup(YsName(names.back()));
                        if (names.size() == 2) {
                            reference.qualifier = safe_strdup(YsName(names.front()));
                        }
                        reference.start_pos = location.Start();
                        reference.end_pos = location.End();
                        references.push_back(reference);
                    }
                    return;
                }
            }
        }
    }
    ParsedExpressionIterator::EnumerateChildren(expression, [&](ParsedExpression &child) {
        CollectNativeAtReferences(child, references);
    });
}
#endif

extern "C" YardstickSelectInfo* yardstick_parse_select(const char* sql) {
    auto* result = new YardstickSelectInfo();
    result->items = nullptr;
    result->item_count = 0;
    result->tables = nullptr;
    result->table_count = 0;
    result->primary_table = nullptr;
    result->group_by_cols = nullptr;
    result->group_by_count = 0;
    result->has_group_by = false;
    result->group_by_all = false;
    result->where_clause = nullptr;
    result->error = nullptr;
    result->native_parsed = false;
    result->at_references = nullptr;
    result->at_reference_count = 0;

    if (!sql) {
        result->error = safe_strdup("NULL SQL input");
        return result;
    }

    try {
        Parser parser(YardstickParserOptions());
#if YARDSTICK_GRAMMAR_EXTENSION
        result->native_parsed = ParseNativeYardstickQuery(sql, parser);
#endif
        if (!result->native_parsed) {
            parser.ParseQuery(sql);
        }

        if (parser.statements.empty()) {
            result->error = safe_strdup("No statements parsed");
            return result;
        }

        auto& stmt = parser.statements[0];
        SelectStatement *select_stmt = nullptr;
        std::unordered_set<std::string> enclosing_cte_names;
        if (stmt->type == StatementType::SELECT_STATEMENT) {
            select_stmt = static_cast<SelectStatement*>(stmt.get());
        }
#if YARDSTICK_GRAMMAR_EXTENSION
        if (result->native_parsed && stmt->type == StatementType::INSERT_STATEMENT) {
            auto &insert = stmt->Cast<InsertStatement>();
            if (insert.node) {
                select_stmt = insert.node->select_statement.get();
                for (auto &cte : insert.node->cte_map.map) {
                    enclosing_cte_names.insert(StringUtil::Lower(YsName(cte.first)));
                }
            }
        }
#endif
        if (!select_stmt) {
            if (result->native_parsed) {
                return result; // No root SELECT references to lower.
            }
            result->error = safe_strdup("Not a SELECT statement");
            return result;
        }

        if (!select_stmt->node || select_stmt->node->type != QueryNodeType::SELECT_NODE) {
            if (result->native_parsed) {
                return result; // Set operations have separate SELECT scopes.
            }
            result->error = safe_strdup("Not a simple SELECT node");
            return result;
        }

        auto* select_node = static_cast<SelectNode*>(select_stmt->node.get());
#if YARDSTICK_GRAMMAR_EXTENSION
        if (result->native_parsed) {
            vector<YardstickMeasureReference> references;
            for (auto &expression : select_node->select_list) {
                CollectNativeAtReferences(*expression, references);
            }
            for (auto &expression : select_node->groups.group_expressions) {
                CollectNativeAtReferences(*expression, references);
            }
            for (auto *expression : {select_node->where_clause.get(), select_node->having.get(),
                                     select_node->qualify.get()}) {
                if (expression) {
                    CollectNativeAtReferences(*expression, references);
                }
            }
            ParsedExpressionIterator::EnumerateQueryNodeModifiers(*select_node,
                [&](unique_ptr<ParsedExpression> &expression) {
                    CollectNativeAtReferences(*expression, references);
                });
            if (!references.empty()) {
                result->at_reference_count = references.size();
                result->at_references = new YardstickMeasureReference[references.size()];
                for (idx_t i = 0; i < references.size(); i++) {
                    result->at_references[i] = references[i];
                }
            }
        }
#endif
        size_t from_pos = FindTopLevelFrom(sql);
        if (from_pos == std::string::npos) {
            from_pos = std::strlen(sql);
        }

        // Process SELECT list
        std::vector<YardstickSelectItem> items;
        for (auto& expr : select_node->select_list) {
            YardstickSelectItem item {};
            item.expression_sql = safe_strdup(expr->ToString());
            item.alias = expr->HasAlias() ? safe_strdup(YsName(expr->GetAlias())) : nullptr;

            optional_idx query_location = expr->GetQueryLocation();
            if (query_location.IsValid()) {
                item.start_pos = static_cast<uint32_t>(query_location.GetIndex());
            } else {
                item.start_pos = 0;
            }
            size_t end_pos = FindSelectItemEnd(sql, item.start_pos, from_pos);
            item.end_pos = static_cast<uint32_t>(end_pos);
#if YARDSTICK_GRAMMAR_EXTENSION
            if (result->native_parsed) {
                // Native locations cover the expression itself, excluding its
                // SELECT alias. Replacements leave original alias/trivia intact.
                auto location = expr->GetQueryLocation();
                item.start_pos = location.IsValid() ? location.Start() : 0;
                item.end_pos = location.IsValid() ? location.End() : 0;
                ParsedExpressionIterator::VisitExpressionClass(*expr, ExpressionClass::SUBQUERY,
                    [&](const ParsedExpression &) { item.contains_subquery = true; });
                if (item.contains_subquery) {
                    auto dimensions = NativeSubqueryDimensions(*expr, *select_node);
                    item.subquery_dimension_count = dimensions.size();
                    if (!dimensions.empty()) {
                        item.subquery_dimensions = new const char *[dimensions.size()] {};
                        for (idx_t i = 0; i < dimensions.size(); i++) {
                            item.subquery_dimensions[i] = safe_strdup(dimensions[i]);
                        }
                    }
                }
                if (expr->GetExpressionClass() == ExpressionClass::COLUMN_REF) {
                    auto &names = expr->Cast<ColumnRefExpression>().ColumnNames();
                    if (names.size() == 1 || names.size() == 2) {
                        item.reference_column = safe_strdup(YsName(names.back()));
                        if (names.size() == 2) {
                            item.reference_qualifier = safe_strdup(YsName(names.front()));
                        }
                    }
                }
            }
#endif

            item.is_aggregate = ExpressionContainsAggregate(expr.get());
            item.is_star = expr->GetExpressionClass() == ExpressionClass::STAR;
            item.is_measure_ref = ExpressionContainsMeasureRef(expr.get());

            items.push_back(item);
        }

        if (!items.empty()) {
            result->item_count = items.size();
            result->items = new YardstickSelectItem[result->item_count];
            for (size_t i = 0; i < items.size(); i++) {
                result->items[i] = items[i];
            }
        }

        // Process FROM clause
        std::vector<YardstickTableRef> tables;
        auto cte_names = std::move(enclosing_cte_names);
#if YARDSTICK_GRAMMAR_EXTENSION
        if (result->native_parsed) {
            for (auto &cte : select_node->cte_map.map) {
                cte_names.insert(StringUtil::Lower(YsName(cte.first)));
            }
        }
#endif
        if (select_node->from_table) {
            CollectTablesFromTableRef(select_node->from_table.get(), tables,
                                      result->native_parsed ? &cte_names : nullptr);
        }

        if (!tables.empty()) {
            result->table_count = tables.size();
            result->tables = new YardstickTableRef[result->table_count];
            for (size_t i = 0; i < tables.size(); i++) {
                result->tables[i] = tables[i];
            }
            result->primary_table = safe_strdup(tables[0].table_name);
        }

        // Process GROUP BY
        if (!select_node->groups.group_expressions.empty()) {
            result->has_group_by = true;
            result->group_by_count = select_node->groups.group_expressions.size();
            result->group_by_cols = new const char*[result->group_by_count];

            for (size_t i = 0; i < select_node->groups.group_expressions.size(); i++) {
                result->group_by_cols[i] = safe_strdup(
                    select_node->groups.group_expressions[i]->ToString()
                );
            }
        }

        // Check for GROUP BY ALL via aggregate_handling
        if (select_node->aggregate_handling == AggregateHandling::FORCE_AGGREGATES) {
            result->group_by_all = true;
            result->has_group_by = true;
        }

        // Process WHERE clause
        if (select_node->where_clause) {
            result->where_clause = safe_strdup(select_node->where_clause->ToString());
        }

    } catch (const std::exception& e) {
        result->error = safe_strdup(e.what());
    }

    return result;
}

//=============================================================================
// FFI Implementation: yardstick_free_select_info
//=============================================================================

extern "C" void yardstick_free_select_info(YardstickSelectInfo* info) {
    if (!info) return;

    for (size_t i = 0; i < info->at_reference_count; i++) {
        free(const_cast<char*>(info->at_references[i].column));
        free(const_cast<char*>(info->at_references[i].qualifier));
    }
    delete[] info->at_references;

    for (size_t i = 0; i < info->item_count; i++) {
        free(const_cast<char*>(info->items[i].expression_sql));
        free(const_cast<char*>(info->items[i].alias));
        free(const_cast<char*>(info->items[i].reference_column));
        free(const_cast<char*>(info->items[i].reference_qualifier));
        for (size_t j = 0; j < info->items[i].subquery_dimension_count; j++) {
            free(const_cast<char*>(info->items[i].subquery_dimensions[j]));
        }
        delete[] info->items[i].subquery_dimensions;
    }
    delete[] info->items;

    for (size_t i = 0; i < info->table_count; i++) {
        free(const_cast<char*>(info->tables[i].table_name));
        free(const_cast<char*>(info->tables[i].alias));
    }
    delete[] info->tables;
    free(const_cast<char*>(info->primary_table));

    for (size_t i = 0; i < info->group_by_count; i++) {
        free(const_cast<char*>(info->group_by_cols[i]));
    }
    delete[] info->group_by_cols;

    free(const_cast<char*>(info->where_clause));
    free(const_cast<char*>(info->error));

    delete info;
}

struct SelectAliasEntry {
    ParsedExpression* expression;
    bool has_subquery;
};

using SelectAliasMap = std::unordered_map<std::string, SelectAliasEntry>;

static void AddTableQualifier(TableQualifierSet &qualifiers, const std::string &name) {
    if (!name.empty()) {
        qualifiers.insert(NormalizeAliasName(name));
    }
}

static void CollectTableQualifiers(const TableRef *ref, TableQualifierSet &qualifiers) {
    if (!ref) {
        return;
    }

    AddTableQualifier(qualifiers, YsName(ref->alias));

    switch (ref->type) {
        case TableReferenceType::BASE_TABLE: {
            auto *base = static_cast<const BaseTableRef*>(ref);
            if (ref->alias.empty()) {
                AddTableQualifier(qualifiers, YsBaseTableName(*base));
            }
            break;
        }

        case TableReferenceType::JOIN: {
            auto *join = static_cast<const JoinRef*>(ref);
            CollectTableQualifiers(join->left.get(), qualifiers);
            CollectTableQualifiers(join->right.get(), qualifiers);
            break;
        }

        default:
            break;
    }
}

static bool FindSelectAliasRef(const ParsedExpression &expr, const SelectAliasMap &aliases,
                               const TableQualifierSet &from_qualifiers,
                               SelectAliasMap::const_iterator &alias_entry) {
    if (expr.GetExpressionClass() != ExpressionClass::COLUMN_REF) {
        return false;
    }

    std::string alias_name;
    auto &colref = expr.Cast<ColumnRefExpression>();
    if (!IsPotentialOrderAliasRef(colref, from_qualifiers, alias_name)) {
        return false;
    }

    alias_entry = aliases.find(NormalizeAliasName(alias_name));
    return alias_entry != aliases.end();
}

static bool IsSimpleSelectAliasOrder(
    const ParsedExpression &expr,
    const SelectAliasMap &aliases,
    const TableQualifierSet &from_qualifiers
) {
    SelectAliasMap::const_iterator alias_entry;
    return FindSelectAliasRef(expr, aliases, from_qualifiers, alias_entry);
}

static void EnumerateOrderAliasScopeChildren(
    const ParsedExpression &expr,
    const std::function<void(const ParsedExpression &child)> &callback
) {
    if (expr.GetExpressionClass() == ExpressionClass::SUBQUERY) {
        auto &subquery_expr = expr.Cast<SubqueryExpression>();
        if (auto* child = YsChild(subquery_expr)) {
            callback(*child);
        }
        return;
    }

    ParsedExpressionIterator::EnumerateChildren(expr, callback);
}

static void EnumerateOrderAliasScopeChildren(
    ParsedExpression &expr,
    const std::function<void(unique_ptr<ParsedExpression> &child)> &callback
) {
    if (expr.GetExpressionClass() == ExpressionClass::SUBQUERY) {
        auto &subquery_expr = expr.Cast<SubqueryExpression>();
        auto &child = YsChildRef(subquery_expr);
        if (child) {
            callback(child);
        }
        return;
    }

    ParsedExpressionIterator::EnumerateChildren(expr, callback);
}

static bool ReferencesSubqueryAlias(
    const ParsedExpression &expr,
    const SelectAliasMap &aliases,
    const TableQualifierSet &from_qualifiers
) {
    SelectAliasMap::const_iterator alias_entry;
    if (FindSelectAliasRef(expr, aliases, from_qualifiers, alias_entry) && alias_entry->second.has_subquery) {
        return true;
    }

    bool found = false;
    EnumerateOrderAliasScopeChildren(expr, [&](const ParsedExpression &child) {
        if (!found && ReferencesSubqueryAlias(child, aliases, from_qualifiers)) {
            found = true;
        }
    });
    return found;
}

static bool InlineSelectAliases(
    unique_ptr<ParsedExpression> &expr,
    const SelectAliasMap &aliases,
    const TableQualifierSet &from_qualifiers
) {
    if (!expr) {
        return false;
    }

    SelectAliasMap::const_iterator alias_entry;
    if (FindSelectAliasRef(*expr, aliases, from_qualifiers, alias_entry)) {
        if (!alias_entry->second.has_subquery) {
            return false;
        }
        auto replacement = alias_entry->second.expression->Copy();
        replacement->ClearAlias();
        expr = std::move(replacement);
        return true;
    }

    bool changed = false;
    EnumerateOrderAliasScopeChildren(*expr, [&](unique_ptr<ParsedExpression> &child) {
        if (InlineSelectAliases(child, aliases, from_qualifiers)) {
            changed = true;
        }
    });
    return changed;
}

//=============================================================================
// FFI Implementation: yardstick_inline_order_by_subquery_aliases
//=============================================================================

extern "C" char* yardstick_inline_order_by_subquery_aliases(const char* sql) {
    if (!sql) {
        return nullptr;
    }

    try {
        Parser parser(YardstickParserOptions());
        parser.ParseQuery(sql);
        if (parser.statements.empty()) {
            return nullptr;
        }

        auto& stmt = parser.statements[0];
        if (stmt->type != StatementType::SELECT_STATEMENT) {
            return nullptr;
        }

        auto* select_stmt = static_cast<SelectStatement*>(stmt.get());
        if (!select_stmt->node || select_stmt->node->type != QueryNodeType::SELECT_NODE) {
            return nullptr;
        }

        auto* select_node = static_cast<SelectNode*>(select_stmt->node.get());
        TableQualifierSet from_qualifiers;
        CollectTableQualifiers(select_node->from_table.get(), from_qualifiers);

        SelectAliasMap aliases;
        bool has_subquery_alias = false;
        for (auto& expr : select_node->select_list) {
            if (!expr->HasAlias()) {
                continue;
            }
            bool has_subquery = expr->HasSubquery();
            aliases[NormalizeAliasName(YsName(expr->GetAlias()))] = SelectAliasEntry { expr.get(), has_subquery };
            has_subquery_alias = has_subquery_alias || has_subquery;
        }

        if (!has_subquery_alias || aliases.empty()) {
            return nullptr;
        }

        bool changed = false;
        for (auto& modifier : select_node->modifiers) {
            if (modifier->type != ResultModifierType::ORDER_MODIFIER) {
                continue;
            }

            auto& order_modifier = modifier->Cast<OrderModifier>();
            for (auto& order : order_modifier.orders) {
                if (!order.expression) {
                    continue;
                }
                if (IsSimpleSelectAliasOrder(*order.expression, aliases, from_qualifiers)) {
                    continue;
                }
                if (!ReferencesSubqueryAlias(*order.expression, aliases, from_qualifiers)) {
                    continue;
                }
                changed = InlineSelectAliases(order.expression, aliases, from_qualifiers) || changed;
            }
        }

        if (!changed) {
            return nullptr;
        }
        return safe_strdup(stmt->ToString());
    } catch (...) {
        return nullptr;
    }
}

//=============================================================================
// FFI Implementation: yardstick_parse_expression
//=============================================================================

bool duckdb::IsYardstickStandardAggregate(const string &name) {
    return IsStandardAggregate(name);
}

extern "C" YardstickExpressionInfo* yardstick_parse_expression(const char* expr_str) {
    auto* result = new YardstickExpressionInfo();
    result->sql = nullptr;
    result->aggregate_func = nullptr;
    result->inner_expr = nullptr;
    result->is_aggregate = false;
    result->is_identifier = false;
    result->is_scalar = false;
    result->error = nullptr;

    if (!expr_str) {
        result->error = safe_strdup("NULL expression input");
        return result;
    }

    try {
        auto expressions = Parser::ParseExpressionList(expr_str, YardstickParserOptions());

        if (expressions.empty()) {
            result->error = safe_strdup("No expressions parsed");
            return result;
        }

        auto& expr = expressions[0];

        result->sql = safe_strdup(expr->ToString());
        result->is_identifier = expr->GetExpressionClass() == ExpressionClass::COLUMN_REF;
        result->is_scalar = expr->IsScalar();
        result->is_aggregate = ExpressionContainsAggregate(expr.get());

        // If it's a simple aggregate function, extract the function name and inner expr
        if (expr->GetExpressionClass() == ExpressionClass::FUNCTION) {
            auto* func = static_cast<FunctionExpression*>(expr.get());
            if (IsStandardAggregate(YsFuncName(*func))) {
                result->aggregate_func = safe_strdup(StringUtil::Upper(YsFuncName(*func)));
                auto args = YsArgs(*func);
                if (!args.empty()) {
                    result->inner_expr = safe_strdup(args[0]->ToString());
                }
            }
        }

    } catch (const std::exception& e) {
        result->error = safe_strdup(e.what());
    }

    return result;
}

extern "C" int32_t yardstick_expressions_equal(const char* left, const char* right) {
    if (!left || !right) {
        return 0;
    }
    try {
        auto options = YardstickParserOptions();
        auto left_expressions = Parser::ParseExpressionList(left, options);
        auto right_expressions = Parser::ParseExpressionList(right, options);
        if (left_expressions.size() != 1 || right_expressions.size() != 1) {
            return 0;
        }
        return ParsedExpression::Equals(left_expressions[0], right_expressions[0]) ? 1 : 0;
    } catch (const std::exception &) {
        return 0;
    }
}

#if YARDSTICK_GRAMMAR_EXTENSION
namespace {
struct VisibleFilterScope {
    bool unqualified_is_local = true;
    bool alias_is_local = true;
    bool inner_alias_shadowed = false;
    TableQualifierSet aliases;
    TableQualifierSet lambda_parameters;
};

class VisibleFilterRewriter {
public:
    VisibleFilterRewriter(string local_alias, const ParserOptions &options)
        : local_alias(std::move(local_alias)), options(options) {
    }

    std::unordered_map<string, unique_ptr<ParsedExpression>> dimensions;

    void Expression(unique_ptr<ParsedExpression> &expression, VisibleFilterScope scope) {
        if (!expression) {
            return;
        }
        if (expression->GetExpressionClass() == ExpressionClass::COLUMN_REF) {
            auto &names = expression->Cast<ColumnRefExpression>().ColumnNames();
            if (scope.lambda_parameters.count(StringUtil::Lower(names[0].GetIdentifierName()))) {
                return;
            }
            bool local = names.size() == 1 ? scope.unqualified_is_local &&
                    !scope.aliases.count(StringUtil::Lower(names[0].GetIdentifierName()))
                : names.size() >= 2 && scope.alias_is_local && !local_alias.empty() &&
                  names[names.size() - 2] == Identifier(local_alias);
            if (local) {
                if (scope.inner_alias_shadowed) {
                    throw ParserException("Conflicting recomputation alias in visible filter");
                }
                auto alias = expression->GetAlias();
                auto entry = dimensions.find(StringUtil::Lower(names.back().GetIdentifierName()));
                if (entry == dimensions.end()) {
                    expression = make_uniq<ColumnRefExpression>(names.back(), Identifier("_inner"));
                } else {
                    expression = entry->second->Copy();
                }
                expression->SetAlias(std::move(alias));
            }
            return;
        }
        if (expression->GetExpressionClass() == ExpressionClass::LAMBDA) {
            auto &lambda = expression->Cast<LambdaExpression>();
            if (lambda.GetLambdaSyntaxType() == LambdaSyntaxType::LAMBDA_KEYWORD) {
                string error;
                auto parameters = lambda.ExtractColumnRefExpressions(error);
                if (!error.empty()) throw ParserException(error);
                for (auto &parameter : parameters) {
                    auto &names = parameter.get().Cast<ColumnRefExpression>().ColumnNames();
                    if (names.size() != 1) throw ParserException("Invalid lambda parameter in visible filter");
                    scope.lambda_parameters.insert(StringUtil::Lower(names[0].GetIdentifierName()));
                }
                Expression(lambda.RightMutable(), scope);
                return;
            }
            // Arrow expressions also represent JSON access. Only explicit
            // lambda syntax establishes parameter bindings before binding.
        }
        if (expression->GetExpressionClass() == ExpressionClass::SUBQUERY) {
            auto &subquery = expression->Cast<SubqueryExpression>();
            // IN/ANY's left operand belongs to the containing expression scope.
            Expression(subquery.GetChildMutable(), scope);
            Query(*subquery.SubqueryMutable()->node, scope);
            return;
        }
        ParsedExpressionIterator::EnumerateChildren(*expression,
            [&](unique_ptr<ParsedExpression> &child) { Expression(child, scope); });
    }

    void AddDimension(const char *name, const char *sql) {
        auto expressions = Parser::ParseExpressionList(sql, options);
        if (expressions.size() != 1 || expressions[0]->HasSubquery()) {
            throw ParserException("Unsupported dimension in visible filter");
        }
        ParsedExpressionIterator::VisitExpressionClassMutable(*expressions[0], ExpressionClass::COLUMN_REF,
            [](ParsedExpression &expression) {
                auto &names = expression.Cast<ColumnRefExpression>().ColumnNamesMutable();
                auto column = names.back();
                names = {Identifier("_inner"), std::move(column)};
            });
        dimensions.emplace(StringUtil::Lower(name), std::move(expressions[0]));
    }

private:
    string local_alias;
    const ParserOptions &options;

    void Qualifiers(const TableRef &table, TableQualifierSet &qualifiers) {
        CollectTableQualifiers(&table, qualifiers);
        if (table.type == TableReferenceType::JOIN) {
            auto &join = table.Cast<JoinRef>();
            Qualifiers(*join.left, qualifiers);
            Qualifiers(*join.right, qualifiers);
        } else if (table.type == TableReferenceType::TABLE_FUNCTION && table.alias.empty()) {
            auto &function = table.Cast<TableFunctionRef>();
            if (function.function->GetExpressionClass() == ExpressionClass::FUNCTION) {
                AddTableQualifier(qualifiers, YsFuncName(function.function->Cast<FunctionExpression>()));
            }
        }
    }

    VisibleFilterScope WithBindings(const TableRef &table, VisibleFilterScope scope) {
        if (table.type == TableReferenceType::EMPTY_FROM) return scope;
        scope.unqualified_is_local = false;
        TableQualifierSet qualifiers;
        Qualifiers(table, qualifiers);
        scope.alias_is_local = scope.alias_is_local &&
            !qualifiers.count(NormalizeAliasName(local_alias));
        scope.inner_alias_shadowed = scope.inner_alias_shadowed || qualifiers.count("_inner");
        return scope;
    }

    void Query(QueryNode &query, VisibleFilterScope outer) {
        // CTE bodies cannot see the containing SELECT's FROM bindings.
        for (auto &entry : query.cte_map.map) {
            if (!entry.second->query_node) {
                throw ParserException("Unsupported CTE in visible filter");
            }
            Query(*entry.second->query_node, outer);
        }
        auto scope = outer;
        switch (query.type) {
        case QueryNodeType::SELECT_NODE: {
            auto &select = query.Cast<SelectNode>();
            if (select.from_table && select.from_table->type != TableReferenceType::EMPTY_FROM) {
                scope = WithBindings(*select.from_table, outer);
                // A derived table's own alias is not visible inside its body.
                Table(*select.from_table, outer);
            }
            for (auto &item : select.select_list) {
                Expression(item, scope);
                if (item->HasAlias()) {
                    scope.aliases.insert(StringUtil::Lower(YsName(item->GetAlias())));
                }
            }
            for (auto &group : select.groups.group_expressions) Expression(group, scope);
            Expression(select.where_clause, scope);
            Expression(select.having, scope);
            Expression(select.qualify, scope);
            break;
        }
        case QueryNodeType::SET_OPERATION_NODE:
            for (auto &child : query.Cast<SetOperationNode>().children) Query(*child, outer);
            // Set result modifiers reference output columns, not base rows.
            scope.unqualified_is_local = false;
            break;
        case QueryNodeType::RECURSIVE_CTE_NODE: {
            auto &cte = query.Cast<RecursiveCTENode>();
            Query(*cte.left, outer);
            Query(*cte.right, outer);
            scope.unqualified_is_local = false;
            for (auto &key : cte.key_targets) Expression(key, scope);
            break;
        }
        default:
            throw ParserException("Unsupported query in visible filter");
        }
        ParsedExpressionIterator::EnumerateQueryNodeModifiers(query,
            [&](unique_ptr<ParsedExpression> &expression) { Expression(expression, scope); });
    }

    void Table(TableRef &table, VisibleFilterScope scope) {
        switch (table.type) {
        case TableReferenceType::JOIN: {
            auto &join = table.Cast<JoinRef>();
            Table(*join.left, scope);
            Table(*join.right, WithBindings(*join.left, scope));
            Expression(join.condition, WithBindings(table, scope));
            break;
        }
        case TableReferenceType::SUBQUERY:
            Query(*table.Cast<SubqueryRef>().subquery->node, scope);
            break;
        case TableReferenceType::EXPRESSION_LIST:
            for (auto &row : table.Cast<ExpressionListRef>().values)
                for (auto &expression : row) Expression(expression, scope);
            break;
        case TableReferenceType::TABLE_FUNCTION: {
            auto &function = table.Cast<TableFunctionRef>();
            Expression(function.function, scope);
            if (function.subquery) Query(*function.subquery->node, scope);
            break;
        }
        case TableReferenceType::BASE_TABLE:
        case TableReferenceType::EMPTY_FROM:
            break;
        default:
            throw ParserException("Unsupported table in visible filter");
        }
    }
};
} // namespace
#endif

extern "C" char* yardstick_rewrite_visible_filter(const char* expression, const char* local_alias,
                                                  const char* const* dimension_names,
                                                  const char* const* dimension_expressions,
                                                  size_t dimension_count, char** error) {
    if (error) *error = nullptr;
#if YARDSTICK_GRAMMAR_EXTENSION
    auto *options = CurrentNativeYardstickParserOptions();
    if (!options || !expression) return nullptr;
    try {
        auto expressions = Parser::ParseExpressionList(expression, *options);
        if (expressions.size() != 1) {
            throw ParserException("Expected one expression in visible filter");
        }
        VisibleFilterRewriter rewriter(local_alias ? local_alias : "", *options);
        for (size_t i = 0; i < dimension_count; i++) {
            rewriter.AddDimension(dimension_names[i], dimension_expressions[i]);
        }
        rewriter.Expression(expressions[0], {});
        return safe_strdup(expressions[0]->ToString());
    } catch (const std::exception &exception) {
        if (error) *error = safe_strdup(exception.what());
        return nullptr;
    }
#endif
    return nullptr;
}

//=============================================================================
// FFI Implementation: yardstick_free_expression_info
//=============================================================================

extern "C" void yardstick_free_expression_info(YardstickExpressionInfo* info) {
    if (!info) return;

    free(const_cast<char*>(info->sql));
    free(const_cast<char*>(info->aggregate_func));
    free(const_cast<char*>(info->inner_expr));
    free(const_cast<char*>(info->error));

    delete info;
}

//=============================================================================
// FFI Implementation: yardstick_parse_create_view
//=============================================================================

extern "C" YardstickCreateViewInfo* yardstick_parse_create_view(const char* sql) {
#if YARDSTICK_GRAMMAR_EXTENSION
    if (auto *native = FindNativeYardstickMeasures(sql)) {
        return native;
    }
#endif
    auto* result = new YardstickCreateViewInfo();
    result->native_parsed = false;
    result->metadata_query_sql = nullptr;
    result->requires_binding = false;
    result->is_measure_view = false;
    result->view_name = nullptr;
    result->clean_sql = nullptr;
    result->measures = nullptr;
    result->measure_count = 0;
    result->error = nullptr;

    if (!sql) {
        result->error = safe_strdup("NULL SQL input");
        return result;
    }

    try {
        // Parse the CREATE VIEW statement
        Parser parser(YardstickParserOptions());
        parser.ParseQuery(sql);

        if (parser.statements.empty()) {
            result->error = safe_strdup("No statements parsed");
            return result;
        }

        // For now, we just check if it's a CREATE VIEW
        // The AS MEASURE detection will need custom logic since DuckDB
        // doesn't natively support this syntax
        auto& stmt = parser.statements[0];

        // Check if statement type is CREATE_VIEW
        if (stmt->type == StatementType::CREATE_STATEMENT) {
            // This is a basic implementation - the full AS MEASURE parsing
            // would need to be done by preprocessing the SQL string
            // or using a custom parser extension

            // For now, signal that this might need preprocessing
            result->is_measure_view = false;
            result->clean_sql = safe_strdup(sql);
        } else {
            result->error = safe_strdup("Not a CREATE VIEW statement");
        }

    } catch (const std::exception& e) {
        result->error = safe_strdup(e.what());
    }

    return result;
}

//=============================================================================
// FFI Implementation: yardstick_free_create_view_info
//=============================================================================

extern "C" void yardstick_free_create_view_info(YardstickCreateViewInfo* info) {
    if (!info) return;

    free(const_cast<char*>(info->view_name));
    free(const_cast<char*>(info->metadata_query_sql));
    free(const_cast<char*>(info->clean_sql));

    for (size_t i = 0; i < info->measure_count; i++) {
        free(const_cast<char*>(info->measures[i].column_name));
        free(const_cast<char*>(info->measures[i].alias_sql));
        free(const_cast<char*>(info->measures[i].expression));
        free(const_cast<char*>(info->measures[i].aggregate_func));
    }
    delete[] info->measures;

    free(const_cast<char*>(info->error));

    delete info;
}

//=============================================================================
// FFI Implementation: yardstick_apply_replacements
//=============================================================================

extern "C" char* yardstick_apply_replacements(
    const char* sql,
    const YardstickReplacement* replacements,
    size_t count
) {
    if (!sql) return nullptr;
    if (count == 0 || !replacements) return safe_strdup(sql);

    std::string result(sql);

    // Sort replacements by start position (descending) to apply from end to start
    std::vector<const YardstickReplacement*> sorted;
    for (size_t i = 0; i < count; i++) {
        sorted.push_back(&replacements[i]);
    }
    std::sort(sorted.begin(), sorted.end(),
        [](const YardstickReplacement* a, const YardstickReplacement* b) {
            return a->start_pos > b->start_pos;
        });

    // Apply replacements from end to start (so positions don't shift)
    for (auto* rep : sorted) {
        if (rep->start_pos <= result.size() && rep->end_pos <= result.size() &&
            rep->start_pos <= rep->end_pos) {
            result.replace(rep->start_pos, rep->end_pos - rep->start_pos,
                           rep->replacement ? rep->replacement : "");
        }
    }

    return safe_strdup(result);
}

//=============================================================================
// FFI Implementation: yardstick_replace_range
//=============================================================================

extern "C" char* yardstick_replace_range(
    const char* sql,
    uint32_t start,
    uint32_t end,
    const char* replacement
) {
    if (!sql) return nullptr;

    std::string result(sql);

    if (start <= result.size() && end <= result.size() && start <= end) {
        result.replace(start, end - start, replacement ? replacement : "");
    }

    return safe_strdup(result);
}

extern "C" char* yardstick_qualify_expression(const char* expr_str, const char* qualifier) {
    if (!expr_str || !qualifier) return nullptr;

    try {
        auto expressions = Parser::ParseExpressionList(expr_str, YardstickParserOptions());
        if (expressions.empty()) {
            return safe_strdup(expr_str);
        }

        for (auto& expr : expressions) {
            QualifyColumnRefs(expr.get(), qualifier);
        }

        if (expressions.size() == 1) {
            return safe_strdup(expressions[0]->ToString());
        }

        std::string result;
        for (size_t i = 0; i < expressions.size(); i++) {
            if (i > 0) result += ", ";
            result += expressions[i]->ToString();
        }
        return safe_strdup(result);
    } catch (const std::exception&) {
        return nullptr;
    }
}

//=============================================================================
// FFI Implementation: yardstick_free_string
//=============================================================================

extern "C" void yardstick_free_string(char* ptr) {
    free(ptr);
}

//=============================================================================
// FFI Implementation: yardstick_expand_aggregate_call
//=============================================================================

extern "C" char* yardstick_expand_aggregate_call(
    const char* measure_name,
    const char* agg_func,
    const YardstickAtModifier* modifiers,
    size_t modifier_count,
    const char* table_name,
    const char* outer_alias,
    const char* outer_where,
    const char* const* group_by_cols,
    size_t group_by_count
) {
    if (!measure_name || !agg_func || !table_name) {
        return nullptr;
    }

    std::string result;

    // Build the subquery for the aggregate
    // Basic form: (SELECT AGG_FUNC(measure) FROM table WHERE ...)

    result = "(SELECT ";
    result += agg_func;
    result += "(";
    result += measure_name;
    result += ") FROM ";
    result += table_name;

    // Apply AT modifiers to build WHERE clause
    std::vector<std::string> where_conditions;

    for (size_t i = 0; i < modifier_count; i++) {
        const auto& mod = modifiers[i];

        switch (mod.type) {
            case YARDSTICK_AT_SET:
                if (mod.dimension && mod.value) {
                    where_conditions.push_back(
                        std::string(mod.dimension) + " = " + std::string(mod.value)
                    );
                }
                break;

            case YARDSTICK_AT_WHERE:
                if (mod.value) {
                    where_conditions.push_back(std::string(mod.value));
                }
                break;

            case YARDSTICK_AT_ALL_GLOBAL:
                // No additional filter for global total
                break;

            case YARDSTICK_AT_ALL_DIM:
                // Remove this dimension from grouping
                // This affects the outer GROUP BY, not the WHERE clause
                break;

            case YARDSTICK_AT_VISIBLE:
                // Apply same filters as outer query
                if (outer_where) {
                    where_conditions.push_back(std::string(outer_where));
                }
                break;

            case YARDSTICK_AT_NONE:
            default:
                break;
        }
    }

    // Add correlation with outer query if we have group by columns
    for (size_t i = 0; i < group_by_count; i++) {
        if (group_by_cols[i]) {
            std::string col = group_by_cols[i];
            std::string correlation;

            if (outer_alias) {
                correlation = std::string(table_name) + "." + col + " IS NOT DISTINCT FROM " +
                              std::string(outer_alias) + "." + col;
            } else {
                // Use the column directly for correlation
                correlation = std::string(table_name) + "." + col + " IS NOT DISTINCT FROM " + col;
            }

            where_conditions.push_back(correlation);
        }
    }

    if (!where_conditions.empty()) {
        result += " WHERE ";
        for (size_t i = 0; i < where_conditions.size(); i++) {
            if (i > 0) result += " AND ";
            result += where_conditions[i];
        }
    }

    result += ")";

    return safe_strdup(result);
}
