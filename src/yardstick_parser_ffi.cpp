/**
 * yardstick_parser_ffi.cpp - C FFI implementation using DuckDB's native parser
 *
 * This file provides FFI functions that use DuckDB's parser to analyze SQL
 * and return results as C structs that can be consumed by Rust.
 *
 * Flow: Rust code -> calls C FFI functions -> C++ uses duckdb::Parser -> returns C structs to Rust
 */

#include "yardstick_ffi.h"
#include "yardstick_parser_compat.hpp"
#include "yardstick_compat.hpp"
#include "frontend_peg.hpp"
#include "aggregate_decorations.hpp"
#include "measure_windows.hpp"
#if YARDSTICK_GRAMMAR_EXTENSION
#include "duckdb/parser/peg/compiled_grammar.hpp"
#include "duckdb/parser/statement/insert_statement.hpp"
#include "duckdb/parser/query_node/set_operation_node.hpp"
#include "duckdb/parser/query_node/recursive_cte_node.hpp"
#include "duckdb/parser/tableref/list.hpp"
#include "duckdb/parser/tableref/table_function_ref.hpp"
#include "duckdb/parser/tableref/expressionlistref.hpp"
#include "duckdb/parser/expression/lambda_expression.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/scalar_function_catalog_entry.hpp"
#include "duckdb/planner/bound_statement.hpp"
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
    return YardstickBuiltinParserOptions();
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

#if YARDSTICK_GRAMMAR_EXTENSION
        case TableReferenceType::PIVOT:
            CollectTablesFromTableRef(ref->Cast<PivotRef>().source.get(), tables, cte_names);
            break;
#endif

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
            if (op->GetExpressionType() == ExpressionType::GROUPING_FUNCTION) {
                return true;
            }
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
    if (!options || !options->compiled_grammar || !options->compiled_grammar->GetRule("YardstickAtModifier")) {
        return -1;
    }
    try {
        auto predicates = YardstickParseExpressionList(predicate, *options);
        auto dimensions = YardstickParseExpressionList(dimension, *options);
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
        delete[] list->scopes[i].cte_definitions;
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
            result->calls = new YardstickAggregateCall[result->count] {};

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
        free(const_cast<char*>(call.call_sql));
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

extern "C" char* yardstick_decorate_measure(
    const char* expression, const char* call_sql, const char* const* dimension_names,
    const char* const* dimension_expressions, size_t dimension_count,
    const char* const* qualifiers, size_t qualifier_count,
    const char* const* binding_ctes, size_t binding_cte_count, char** error) {
    if (error) *error = nullptr;
#if YARDSTICK_GRAMMAR_EXTENSION
    try {
        vector<pair<string, string>> dimensions;
        for (size_t i = 0; i < dimension_count; i++) {
            dimensions.emplace_back(dimension_names[i], dimension_expressions[i]);
        }
        vector<string> local_qualifiers;
        for (size_t i = 0; i < qualifier_count; i++) local_qualifiers.emplace_back(qualifiers[i]);
        auto options = YardstickParserOptions();
        vector<string> cte_queries;
        for (size_t i = 0; i < binding_cte_count; i++) cte_queries.emplace_back(binding_ctes[i]);
        NativeYardstickCteBindScope binding_scope(cte_queries, options);
        auto calls = YardstickParseExpressionList(call_sql, options);
        if (calls.size() != 1) throw ParserException("Expected one AGGREGATE call");
        // Window FILTER selects input rows in the lineage stage. DISTINCT and
        // argument ordering apply to the aggregate leaves being recomputed.
        const bool is_window = calls[0]->GetExpressionClass() == ExpressionClass::WINDOW;
        if (is_window) {
            auto &window = calls[0]->Cast<WindowExpression>();
            vector<unique_ptr<ParsedExpression>> arguments;
            for (auto &argument : window.GetArguments()) arguments.push_back(argument.GetExpression().Copy());
            auto orders = make_uniq<OrderModifier>();
            for (auto &order : window.ArgOrders()) {
                orders->orders.emplace_back(order.type, order.null_order, order.expression->Copy());
            }
            calls[0] = make_uniq<FunctionExpression>(Identifier("aggregate"), std::move(arguments),
                nullptr, std::move(orders), window.Distinct());
        }
        return safe_strdup(DecorateYardstickMeasureExpression(
            expression, calls[0]->ToString(), dimensions, local_qualifiers, options, is_window));
    } catch (const std::exception &exception) {
        if (error) *error = safe_strdup(exception.what());
    }
#endif
    return nullptr;
}

extern "C" char* yardstick_window_marker(const char* call_sql, const char* marker_name, char** error) {
    if (error) *error = nullptr;
#if YARDSTICK_GRAMMAR_EXTENSION
    try {
        auto expressions = YardstickParseExpressionList(call_sql, YardstickParserOptions());
        if (expressions.size() != 1 || expressions[0]->GetExpressionClass() != ExpressionClass::WINDOW) {
            throw ParserException("Expected one windowed AGGREGATE call");
        }
        auto &window = expressions[0]->Cast<WindowExpression>();
        window.SetFunctionName(marker_name);
        window.GetArgumentsMutable().clear();
        window.GetArgumentsMutable().emplace_back(ConstantExpression::FromValue(Value::INTEGER(0)));
        window.DistinctMutable() = false;
        return safe_strdup(window.ToString());
    } catch (const std::exception &exception) {
        if (error) *error = safe_strdup(exception.what());
    }
#endif
    return nullptr;
}

extern "C" char* yardstick_rewrite_measure_windows(
    const char* sql, const YardstickWindowSource* sources, size_t source_count,
    const YardstickWindowCall* calls, size_t call_count,
    const char* const* visible_ctes, size_t visible_cte_count,
    const char* const* binding_ctes, size_t binding_cte_count, char** error) {
    if (error) *error = nullptr;
#if YARDSTICK_GRAMMAR_EXTENSION
    try {
        vector<MeasureWindowSource> source_info;
        for (size_t i = 0; i < source_count; i++) {
            auto &source = sources[i];
            MeasureWindowSource info {source.key, source.relation_name, source.alias, source.clean_select_sql,
                                      source.grouped, {}};
            for (size_t j = 0; j < source.dimension_count; j++) {
                info.dimensions.emplace_back(source.dimension_names[j], source.dimension_expressions[j]);
            }
            source_info.push_back(std::move(info));
        }
        vector<MeasureWindowCall> call_info;
        for (size_t i = 0; i < call_count; i++) {
            auto &call = calls[i];
            MeasureWindowCall info {call.marker_name, call.source_key, call.expression_sql, {}};
            for (size_t j = 0; j < call.modifier_count; j++) {
                auto &modifier = call.modifiers[j];
                WindowContextType type;
                switch (modifier.type) {
                case YARDSTICK_AT_ALL_GLOBAL: type = WindowContextType::ALL_GLOBAL; break;
                case YARDSTICK_AT_ALL_DIM: type = WindowContextType::ALL; break;
                case YARDSTICK_AT_SET: type = WindowContextType::SET; break;
                case YARDSTICK_AT_WHERE: type = WindowContextType::WHERE; break;
                case YARDSTICK_AT_VISIBLE: type = WindowContextType::VISIBLE; break;
                default: continue;
                }
                info.modifiers.push_back({type, modifier.dimension ? modifier.dimension : "",
                                         modifier.value ? modifier.value : ""});
            }
            call_info.push_back(std::move(info));
        }
        vector<string> cte_names;
        for (size_t i = 0; i < visible_cte_count; i++) {
            cte_names.push_back(visible_ctes[i]);
        }
        vector<string> cte_queries;
        for (size_t i = 0; i < binding_cte_count; i++) {
            cte_queries.push_back(binding_ctes[i]);
        }
        return safe_strdup(RewriteNativeMeasureWindows(sql, source_info, call_info, cte_names,
                                                      cte_queries, YardstickParserOptions()));
    } catch (const std::exception &exception) {
        if (error) *error = safe_strdup(exception.what());
    }
#endif
    return nullptr;
}

//=============================================================================
// FFI Implementation: yardstick_parse_select
//=============================================================================

#if YARDSTICK_GRAMMAR_EXTENSION
static std::unordered_set<string> SelectInputColumns(SelectNode &select) {
    std::unordered_set<string> names;
    if (!CurrentNativeYardstickClientContext() || !select.from_table ||
        select.from_table->type == TableReferenceType::EMPTY_FROM) return names;
    auto probe = select.Copy();
    auto &layout = probe->Cast<SelectNode>();
    layout.select_list.clear();
    layout.select_list.push_back(make_uniq<StarExpression>());
    layout.groups = GroupByNode();
    layout.where_clause.reset();
    layout.having.reset();
    layout.qualify.reset();
    layout.modifiers.clear();
    layout.aggregate_handling = AggregateHandling::STANDARD_HANDLING;
    auto bound = BindNativeYardstickProbe(*probe);
    for (auto &name : bound.names) names.insert(StringUtil::Lower(name.GetIdentifierName()));
    return names;
}

static unique_ptr<ParsedExpression> ResolveGroupingDimension(
    SelectNode &select, const ParsedExpression &dimension, const std::unordered_set<string> &input_columns) {
    const ParsedExpression *resolved = &dimension;
    if (dimension.GetExpressionClass() == ExpressionClass::CONSTANT) {
        auto value = dimension.Cast<ConstantExpression>().GetLiteral().ToValue();
        if (!value.IsNull() && value.type().IsIntegral()) {
            auto ordinal = value.GetValue<int64_t>();
            if (ordinal > 0 && idx_t(ordinal) <= select.select_list.size()) {
                resolved = select.select_list[ordinal - 1].get();
            }
        }
    } else if (dimension.GetExpressionClass() == ExpressionClass::COLUMN_REF) {
        auto &column = dimension.Cast<ColumnRefExpression>();
        auto name = StringUtil::Lower(YsName(column.GetColumnName()));
        if (!column.IsQualified() && !input_columns.count(name)) {
            for (auto &projection : select.select_list) {
                if (projection->HasAlias() && StringUtil::Lower(YsName(projection->GetAlias())) == name) {
                    resolved = projection.get();
                    break;
                }
            }
        }
    }
    auto result = resolved->Copy();
    result->ClearAlias();
    return result;
}

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

static vector<string> NativeSubqueryDimensions(
    ParsedExpression &projection, SelectNode &outer,
    const std::function<void(ColumnRefExpression &)> &rewrite = {}) {
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
                if (rewrite) {
                    rewrite(expr.Cast<ColumnRefExpression>());
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
        std::unordered_set<string> input_columns;
#if YARDSTICK_GRAMMAR_EXTENSION
        if (result->native_parsed && CurrentNativeYardstickClientContext()) {
            input_columns = SelectInputColumns(*select_node);
        }
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
        std::unordered_set<std::string> measure_aliases;
        std::function<bool(const ParsedExpression &)> references_measure_alias =
            [&](const ParsedExpression &expression) {
                // Nested queries own their aliases and column references.
                if (expression.GetExpressionClass() == ExpressionClass::SUBQUERY) {
                    return false;
                }
                if (expression.GetExpressionClass() == ExpressionClass::COLUMN_REF) {
                    auto &column = expression.Cast<ColumnRefExpression>();
                    return YsColumnNameCount(column) == 1 &&
                           !input_columns.count(StringUtil::Lower(YsName(column.GetColumnName()))) &&
                           measure_aliases.count(StringUtil::Lower(YsName(column.GetColumnName())));
                }
                bool found = false;
                ParsedExpressionIterator::EnumerateChildren(expression, [&](const ParsedExpression &child) {
                    found |= references_measure_alias(child);
                });
                return found;
            };
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
                // Indirection nodes can cover only their suffix: region[1:1]
                // places the slice's root location on [1:1]. Include this
                // expression's operands, keeping SELECT aliases and neighboring
                // projections outside the replacement range.
                auto source_length = std::strlen(sql);
                QueryLocation expression_location;
                std::function<void(const ParsedExpression &)> include_span =
                    [&](const ParsedExpression &node) {
                    auto location = node.GetQueryLocation();
                    if (location.IsValid() && location.End() > location.Start() &&
                        location.End() <= source_length) {
                        expression_location = expression_location.Merge(location);
                    }
                    ParsedExpressionIterator::EnumerateChildren(node, include_span);
                };
                include_span(*expr);
                item.start_pos = expression_location.IsValid() ? expression_location.Start() : 0;
                item.end_pos = expression_location.IsValid() ? expression_location.End() : 0;
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
            item.contains_window = expr->IsWindow();
            item.is_star = expr->GetExpressionClass() == ExpressionClass::STAR;
            item.is_measure_ref = ExpressionContainsMeasureRef(expr.get()) || references_measure_alias(*expr);
            if (expr->HasAlias() && (item.is_measure_ref || item.is_aggregate)) {
                measure_aliases.insert(StringUtil::Lower(YsName(expr->GetAlias())));
            }

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
#if YARDSTICK_GRAMMAR_EXTENSION
                if (result->native_parsed) {
                    result->group_by_cols[i] = safe_strdup(ResolveGroupingDimension(
                        *select_node, *select_node->groups.group_expressions[i], input_columns)->ToString());
                    continue;
                }
#endif
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

extern "C" YardstickSelectInfo* yardstick_parse_select_with_ctes(
    const char* sql, const char* const* binding_ctes, size_t binding_cte_count) {
    if (binding_cte_count == 0) return yardstick_parse_select(sql);
#if YARDSTICK_GRAMMAR_EXTENSION
    try {
        vector<string> queries;
        for (size_t i = 0; i < binding_cte_count; i++) queries.emplace_back(binding_ctes[i]);
        NativeYardstickCteBindScope scope(queries, YardstickParserOptions());
        return yardstick_parse_select(sql);
    } catch (const std::exception &error) {
        auto *result = new YardstickSelectInfo {};
        result->error = safe_strdup(error.what());
        return result;
    }
#else
    return yardstick_parse_select(sql);
#endif
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
    const TableQualifierSet &from_qualifiers,
    const std::unordered_set<string> *input_columns = nullptr
) {
    if (!expr) {
        return false;
    }

    SelectAliasMap::const_iterator alias_entry;
    if (FindSelectAliasRef(*expr, aliases, from_qualifiers, alias_entry)) {
        if (input_columns && !expr->Cast<ColumnRefExpression>().IsQualified() &&
            input_columns->count(alias_entry->first)) return false;
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
        if (InlineSelectAliases(child, aliases, from_qualifiers, input_columns)) {
            changed = true;
        }
    });
    return changed;
}

extern "C" bool yardstick_requires_select_alias_binding(const char* sql) {
#if YARDSTICK_GRAMMAR_EXTENSION
    try {
        Parser parser(YardstickParserOptions());
        if (!ParseNativeYardstickQuery(sql, parser)) return false;
        if (parser.statements.size() != 1 ||
            parser.statements[0]->type != StatementType::SELECT_STATEMENT) return false;
        auto &statement = parser.statements[0]->Cast<SelectStatement>();
        if (statement.node->type != QueryNodeType::SELECT_NODE) return false;
        auto &select = statement.node->Cast<SelectNode>();
        TableQualifierSet qualifiers;
        CollectTableQualifiers(select.from_table.get(), qualifiers);
        SelectAliasMap aliases;
        std::function<bool(const ParsedExpression &)> references_alias = [&](const ParsedExpression &expression) {
            SelectAliasMap::const_iterator entry;
            if (FindSelectAliasRef(expression, aliases, qualifiers, entry)) return true;
            bool found = false;
            EnumerateOrderAliasScopeChildren(expression, [&](const ParsedExpression &child) {
                found |= references_alias(child);
            });
            return found;
        };
        for (auto &expression : select.select_list) {
            if (references_alias(*expression)) return true;
            if (expression->HasAlias()) {
                aliases[NormalizeAliasName(YsName(expression->GetAlias()))] = {expression.get(), false};
            }
        }
        for (auto &expression : select.groups.group_expressions) {
            if (references_alias(*expression)) return true;
        }
        return select.having && references_alias(*select.having);
    } catch (const std::exception &) {
        return false;
    }
#endif
    return false;
}

//=============================================================================
// FFI Implementation: yardstick_inline_order_by_subquery_aliases
//=============================================================================

#if YARDSTICK_GRAMMAR_EXTENSION
static unique_ptr<ParsedExpression> GroupingContextExpression(const ParsedExpression &expression,
                                                             bool unqualify_columns = false) {
    auto result = expression.Copy();
    if (result->GetExpressionClass() == ExpressionClass::FUNCTION) {
        auto &function = result->Cast<FunctionExpression>();
        if (StringUtil::CIEquals(YsFuncName(function), "any_value") && YsArgs(function).size() == 1) {
            result = YsArgs(function)[0]->Copy();
        }
    }
    std::function<void(ParsedExpression &)> unqualify = [&](ParsedExpression &node) {
        if (node.GetExpressionClass() == ExpressionClass::COLUMN_REF) {
            auto &names = node.Cast<ColumnRefExpression>().ColumnNamesMutable();
            if (names.size() > 1) {
                auto name = names.back();
                names = {name};
            }
        }
        ParsedExpressionIterator::EnumerateChildren(node, [&](unique_ptr<ParsedExpression> &child) {
            unqualify(*child);
        });
    };
    if (unqualify_columns) unqualify(*result);
    result->ClearAlias();
    return result;
}

// Correlation markers identify only predicates generated for grouping context.
// SET predicates and user-authored subqueries must retain their own semantics.
static bool RewriteGroupingCorrelations(unique_ptr<ParsedExpression> &expression,
                                        const vector<unique_ptr<ParsedExpression>> &dimensions,
                                        const GroupingSet *grouping_set) {
    if (!expression) return false;
    if (expression->GetExpressionClass() == ExpressionClass::FUNCTION) {
        auto &function = expression->Cast<FunctionExpression>();
        if (YsFuncName(function) == "__yardstick_grouping_correlation" && YsArgs(function).size() == 1) {
            auto condition = YsArgs(function)[0]->Copy();
            bool removed = false;
            if (grouping_set && condition->GetExpressionClass() == ExpressionClass::COMPARISON) {
                auto &comparison = condition->Cast<ComparisonExpression>();
                for (idx_t i = 0; i < dimensions.size(); i++) {
                    bool qualified = false;
                    ParsedExpressionIterator::VisitExpressionClass(*dimensions[i], ExpressionClass::COLUMN_REF,
                        [&](const ParsedExpression &column) {
                            qualified |= column.Cast<ColumnRefExpression>().IsQualified();
                        });
                    auto outer_dimension = GroupingContextExpression(comparison.Right(), !qualified);
                    if (!grouping_set->count(ProjectionIndex(i)) && outer_dimension->Equals(*dimensions[i])) {
                        removed = true;
                        break;
                    }
                }
            }
            expression = removed ? ConstantExpression::FromValue(Value::BOOLEAN(true)) : std::move(condition);
            return removed;
        }
    }
    if (expression->GetExpressionClass() == ExpressionClass::SUBQUERY) {
        auto &subquery = expression->Cast<SubqueryExpression>();
        if (subquery.Subquery() && subquery.Subquery()->node->type == QueryNodeType::SELECT_NODE) {
            auto &select = subquery.SubqueryMutable()->node->Cast<SelectNode>();
            return RewriteGroupingCorrelations(select.where_clause, dimensions, grouping_set);
        }
        return false;
    }
    bool removed = false;
    ParsedExpressionIterator::EnumerateChildren(*expression, [&](unique_ptr<ParsedExpression> &child) {
        removed |= RewriteGroupingCorrelations(child, dimensions, grouping_set);
    });
    return removed;
}

static bool RewriteGroupingSetContexts(SelectNode &select) {
    vector<unique_ptr<ParsedExpression>> dimensions;
    auto input_columns = SelectInputColumns(select);
    for (auto &dimension : select.groups.group_expressions) {
        auto resolved = ResolveGroupingDimension(select, *dimension, input_columns);
        dimensions.push_back(GroupingContextExpression(*resolved));
    }
    bool changed = false;
    std::function<void(unique_ptr<ParsedExpression> &, bool)> rewrite =
        [&](unique_ptr<ParsedExpression> &expression, bool grouped_context) {
        if (!expression || expression->GetExpressionClass() == ExpressionClass::SUBQUERY) return;
        if (expression->GetExpressionClass() == ExpressionClass::FUNCTION) {
            auto &function = expression->Cast<FunctionExpression>();
            if (YsFuncName(function) == "__yardstick_grouping_context" && YsArgs(function).size() == 1) {
                auto original = YsArgs(function)[0]->Copy();
                auto alias = expression->GetAlias();
                auto result = make_uniq<CaseExpression>();
                if (grouped_context && select.groups.grouping_sets.size() > 1) {
                    for (auto &grouping_set : select.groups.grouping_sets) {
                        auto value = original->Copy();
                        if (!RewriteGroupingCorrelations(value, dimensions, &grouping_set)) continue;
                        vector<unique_ptr<ParsedExpression>> conditions;
                        for (idx_t i = 0; i < dimensions.size(); i++) {
                            vector<unique_ptr<ParsedExpression>> arguments;
                            arguments.push_back(dimensions[i]->Copy());
                            auto grouping = make_uniq<OperatorExpression>(ExpressionType::GROUPING_FUNCTION,
                                                                          std::move(arguments));
                            conditions.push_back(make_uniq<ComparisonExpression>(ExpressionType::COMPARE_EQUAL,
                                std::move(grouping), ConstantExpression::FromValue(
                                    Value::BIGINT(grouping_set.count(ProjectionIndex(i)) ? 0 : 1))));
                        }
                        auto condition = make_uniq<ConjunctionExpression>(ExpressionType::CONJUNCTION_AND,
                                                                          std::move(conditions));
                        result->CaseChecksMutable().push_back({std::move(condition), std::move(value)});
                    }
                }
                RewriteGroupingCorrelations(original, dimensions, nullptr);
                if (result->CaseChecks().empty()) {
                    expression = std::move(original);
                } else {
                    // GROUPING executes in the outer grouped query. Moving it
                    // into a correlated subquery is rejected by DuckDB, and
                    // checking NULL would confuse real NULLs with subtotals.
                    result->ElseMutable() = std::move(original);
                    expression = std::move(result);
                }
                expression->SetAlias(alias);
                changed = true;
                return;
            }
        }
        ParsedExpressionIterator::EnumerateChildren(*expression, [&](unique_ptr<ParsedExpression> &child) {
            rewrite(child, grouped_context);
        });
    };
    for (auto &expression : select.select_list) rewrite(expression, true);
    rewrite(select.having, true);
    rewrite(select.qualify, true);
    // WHERE runs before grouping. Its measure contexts use individual rows,
    // even when the enclosing SELECT also produces grouping-set subtotals.
    rewrite(select.where_clause, false);
    std::function<void(TableRef &)> rewrite_row_predicates = [&](TableRef &table) {
        switch (table.type) {
        case TableReferenceType::JOIN: {
            auto &join = table.Cast<JoinRef>();
            rewrite_row_predicates(*join.left);
            rewrite_row_predicates(*join.right);
            rewrite(join.condition, false);
            break;
        }
        case TableReferenceType::TABLE_FUNCTION:
            rewrite(table.Cast<TableFunctionRef>().function, false);
            break;
        case TableReferenceType::EXPRESSION_LIST:
            for (auto &row : table.Cast<ExpressionListRef>().values) {
                for (auto &expression : row) rewrite(expression, false);
            }
            break;
        default:
            // Derived tables own their grouping context and are lowered as
            // separate native query scopes.
            break;
        }
    };
    if (select.from_table) rewrite_row_predicates(*select.from_table);
    ParsedExpressionIterator::EnumerateQueryNodeModifiers(select,
        [&](unique_ptr<ParsedExpression> &expression) { rewrite(expression, true); });
    return changed;
}
#endif

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
        bool changed = false;
#if YARDSTICK_GRAMMAR_EXTENSION
        changed = RewriteGroupingSetContexts(*select_node);
#endif
        TableQualifierSet from_qualifiers;
        CollectTableQualifiers(select_node->from_table.get(), from_qualifiers);
        std::unordered_set<string> input_columns;
        bool bind_select_aliases = true;
#if YARDSTICK_GRAMMAR_EXTENSION
        input_columns = SelectInputColumns(*select_node);
        bind_select_aliases = CurrentNativeYardstickClientContext() != nullptr;
#endif

        SelectAliasMap aliases;
        bool has_subquery_alias = false;
        for (auto& expr : select_node->select_list) {
            // DuckDB cannot reuse a scalar-subquery alias in another SELECT
            // expression. Expand references in declaration order so dependent
            // aliases already contain their fully lowered definitions.
            if (bind_select_aliases && ReferencesSubqueryAlias(*expr, aliases, from_qualifiers)) {
                auto alias = expr->GetAlias();
                changed = InlineSelectAliases(expr, aliases, from_qualifiers, &input_columns) || changed;
                expr->SetAlias(alias);
            }
            if (!expr->HasAlias()) {
                continue;
            }
            bool has_subquery = expr->HasSubquery();
            aliases[NormalizeAliasName(YsName(expr->GetAlias()))] = SelectAliasEntry { expr.get(), has_subquery };
            has_subquery_alias = has_subquery_alias || has_subquery;
        }

        if (!has_subquery_alias || aliases.empty()) {
            return changed ? safe_strdup(stmt->ToString()) : nullptr;
        }

        if (bind_select_aliases && select_node->having &&
            ReferencesSubqueryAlias(*select_node->having, aliases, from_qualifiers)) {
            changed = InlineSelectAliases(select_node->having, aliases, from_qualifiers, &input_columns) || changed;
        }
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

extern "C" char* yardstick_inline_order_by_subquery_aliases_with_ctes(
    const char* sql, const char* const* binding_ctes, size_t binding_cte_count) {
    if (binding_cte_count == 0) return yardstick_inline_order_by_subquery_aliases(sql);
#if YARDSTICK_GRAMMAR_EXTENSION
    try {
        vector<string> queries;
        for (size_t i = 0; i < binding_cte_count; i++) queries.emplace_back(binding_ctes[i]);
        NativeYardstickCteBindScope scope(queries, YardstickParserOptions());
        return yardstick_inline_order_by_subquery_aliases(sql);
    } catch (const std::exception &) {
        return nullptr;
    }
#else
    return yardstick_inline_order_by_subquery_aliases(sql);
#endif
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
        auto expressions = YardstickParseExpressionList(expr_str, YardstickParserOptions());

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
        auto left_expressions = YardstickParseExpressionList(left, options);
        auto right_expressions = YardstickParseExpressionList(right, options);
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
        auto expressions = YardstickParseExpressionList(sql, options);
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

#if YARDSTICK_GRAMMAR_EXTENSION
// De Morgan, distributivity, idempotence, and absorption hold for SQL's
// three-valued AND/OR/NOT. Normalize before separating dimension predicates so
// equivalent NOT and factorable OR forms retain the same independent filters.
static unique_ptr<ParsedExpression> ContextNegationNormalForm(const ParsedExpression &expression, bool negate = false) {
    auto type = expression.GetExpressionType();
    if (type == ExpressionType::OPERATOR_NOT) {
        auto &children = expression.Cast<OperatorExpression>().GetChildren();
        if (children.size() == 1) return ContextNegationNormalForm(*children[0], !negate);
    }
    if (type == ExpressionType::CONJUNCTION_AND || type == ExpressionType::CONJUNCTION_OR) {
        if (negate) type = type == ExpressionType::CONJUNCTION_AND
            ? ExpressionType::CONJUNCTION_OR : ExpressionType::CONJUNCTION_AND;
        vector<unique_ptr<ParsedExpression>> children;
        for (auto &child : expression.Cast<ConjunctionExpression>().GetChildren()) {
            children.push_back(ContextNegationNormalForm(*child, negate));
        }
        return make_uniq<ConjunctionExpression>(type, std::move(children));
    }
    auto result = expression.Copy();
    if (!negate) return result;
    return make_uniq<OperatorExpression>(ExpressionType::OPERATOR_NOT, std::move(result));
}

using ContextClause = vector<unique_ptr<ParsedExpression>>;
using ContextConjunction = vector<ContextClause>;

static bool ContextExpressionCanRepeat(const ParsedExpression &expression) {
    if (expression.GetExpressionClass() == ExpressionClass::SUBQUERY) return false;
    if (expression.GetExpressionClass() == ExpressionClass::FUNCTION) {
        auto *context = CurrentNativeYardstickClientContext();
        if (!context) return false;
        try {
            auto &function = expression.Cast<FunctionExpression>();
            auto entry = Catalog::GetEntry<ScalarFunctionCatalogEntry>(
                *context, function.GetQualifiedName(), OnEntryNotFound::RETURN_NULL);
            if (!entry || entry->functions.functions.empty()) return false;
            for (auto &overload : entry->functions.functions) {
                if (overload->GetStability() == FunctionStability::VOLATILE) return false;
            }
        } catch (const std::exception &) {
            return false; // No catalog proof permits no evaluation-changing rewrite.
        }
    }
    bool repeatable = true;
    ParsedExpressionIterator::EnumerateChildren(expression, [&](const ParsedExpression &child) {
        repeatable &= ContextExpressionCanRepeat(child);
    });
    return repeatable;
}

static ContextConjunction ContextConjunctiveNormalForm(const ParsedExpression &expression) {
    constexpr idx_t MAX_CONTEXT_CLAUSES = 128;
    auto type = expression.GetExpressionType();
    if (type != ExpressionType::CONJUNCTION_AND && type != ExpressionType::CONJUNCTION_OR) {
        ContextClause clause;
        clause.push_back(expression.Copy());
        ContextConjunction result;
        result.push_back(std::move(clause));
        return result;
    }
    ContextConjunction result;
    for (auto &child : expression.Cast<ConjunctionExpression>().GetChildren()) {
        auto next = ContextConjunctiveNormalForm(*child);
        if (result.empty()) {
            result = std::move(next);
        } else if (type == ExpressionType::CONJUNCTION_AND) {
            for (auto &clause : next) result.push_back(std::move(clause));
        } else {
            if (next.size() > MAX_CONTEXT_CLAUSES / result.size()) {
                throw ParserException("AT dimension replacement requires a simpler predicate; use an explicit AT WHERE predicate");
            }
            ContextConjunction product;
            for (auto &left : result) {
                for (auto &right : next) {
                    ContextClause clause;
                    for (auto &atom : left) clause.push_back(atom->Copy());
                    for (auto &atom : right) clause.push_back(atom->Copy());
                    product.push_back(std::move(clause));
                }
            }
            result = std::move(product);
        }
        if (result.size() > MAX_CONTEXT_CLAUSES) {
            throw ParserException("AT dimension replacement requires a simpler predicate; use an explicit AT WHERE predicate");
        }
    }
    return result;
}

static unique_ptr<ParsedExpression> ContextDimensionReference(const ParsedExpression &expression,
                                                             const string &source_alias) {
    auto result = expression.Copy();
    ParsedExpressionIterator::VisitExpressionClassMutable(*result, ExpressionClass::COLUMN_REF,
        [&](ParsedExpression &column) {
            auto &names = column.Cast<ColumnRefExpression>().ColumnNamesMutable();
            if (names.size() == 1 || (names.size() == 2 && names[0] == Identifier(source_alias))) {
                auto name = names.back();
                names = {Identifier("_inner"), std::move(name)};
            }
        });
    return result;
}

static bool ContextReferencesDimension(const ParsedExpression &expression,
                                       const vector<unique_ptr<ParsedExpression>> &dimensions,
                                       const string &source_alias) {
    auto canonical = ContextDimensionReference(expression, source_alias);
    for (auto &dimension : dimensions) {
        if (canonical->Equals(*dimension)) return true;
    }
    if (expression.GetExpressionClass() == ExpressionClass::SUBQUERY) return false;
    bool found = false;
    ParsedExpressionIterator::EnumerateChildren(expression, [&](const ParsedExpression &child) {
        found |= ContextReferencesDimension(child, dimensions, source_alias);
    });
    return found;
}

static void DropContextDimensions(unique_ptr<ParsedExpression> &expression,
                                  const vector<unique_ptr<ParsedExpression>> &excluded,
                                  const string &source_alias = string()) {
    if (!expression || excluded.empty()) return;
    // Unrelated subtrees are already valid context terms. Keep their original
    // shape and repeated evaluations, including separate volatile OR calls.
    if (!ContextReferencesDimension(*expression, excluded, source_alias)) return;
    if (!ContextExpressionCanRepeat(*expression)) {
        if (expression->GetExpressionType() == ExpressionType::CONJUNCTION_AND) {
            ParsedExpressionIterator::EnumerateChildren(*expression, [&](unique_ptr<ParsedExpression> &child) {
                DropContextDimensions(child, excluded, source_alias);
            });
            return;
        }
        bool boolean_reassociation = false;
        ParsedExpressionIterator::VisitExpressionClass(*expression, ExpressionClass::CONJUNCTION,
            [&](const ParsedExpression &) { boolean_reassociation = true; });
        if (boolean_reassociation) {
            throw ParserException("AT dimension replacement cannot safely normalize a volatile or unproven predicate; "
                                  "use an explicit AT WHERE predicate");
        }
    }
    auto normalized = ContextNegationNormalForm(*expression);
    auto clauses = ContextConjunctiveNormalForm(*normalized);
    // Deduplicate disjuncts, then absorb clauses containing a smaller clause.
    // For example (year=A AND region=N) OR (year=B AND region=N)
    // reduces to (year=A OR year=B) AND region=N before removing year.
    for (auto &clause : clauses) {
        for (idx_t i = 0; i < clause.size(); i++) {
            for (idx_t j = i + 1; j < clause.size();) {
                if (clause[i]->Equals(*clause[j])) clause.erase(clause.begin() + j);
                else j++;
            }
        }
    }
    vector<bool> absorbed(clauses.size(), false);
    for (idx_t i = 0; i < clauses.size(); i++) {
        for (idx_t j = 0; j < clauses.size(); j++) {
            if (i == j || clauses[i].size() > clauses[j].size() ||
                (clauses[i].size() == clauses[j].size() && i > j)) continue;
            bool subset = true;
            for (auto &atom : clauses[i]) {
                bool present = false;
                for (auto &candidate : clauses[j]) present |= atom->Equals(*candidate);
                subset &= present;
            }
            if (subset) absorbed[j] = true;
        }
    }
    std::function<void(const ParsedExpression &, bool &, bool &)> dependencies;
    dependencies = [&](const ParsedExpression &term, bool &overridden, bool &other) {
        auto canonical = ContextDimensionReference(term, source_alias);
        for (auto &dimension : excluded) {
            if (canonical->Equals(*dimension)) {
                overridden = true;
                return;
            }
        }
        if (term.GetExpressionClass() == ExpressionClass::COLUMN_REF ||
            term.GetExpressionClass() == ExpressionClass::SUBQUERY) {
            other = true;
            return;
        }
        ParsedExpressionIterator::EnumerateChildren(term, [&](const ParsedExpression &child) {
            dependencies(child, overridden, other);
        });
    };
    vector<unique_ptr<ParsedExpression>> remaining;
    for (idx_t i = 0; i < clauses.size(); i++) {
        if (absorbed[i]) continue;
        bool overridden = false, other = false;
        for (auto &atom : clauses[i]) {
            bool atom_overridden = false, atom_other = false;
            dependencies(*atom, atom_overridden, atom_other);
            if (atom_overridden && atom_other && atom->GetExpressionClass() == ExpressionClass::COMPARISON) {
                // A join equality such as source.year = allowed.year constrains
                // the source year using an external value. It is a year term;
                // the external operand is not another source dimension.
                bool other_source_dimension = false;
                ParsedExpressionIterator::VisitExpressionClass(*atom, ExpressionClass::COLUMN_REF,
                    [&](const ParsedExpression &column) {
                        auto canonical = ContextDimensionReference(column, source_alias);
                        bool removed = false;
                        for (auto &dimension : excluded) removed |= canonical->Equals(*dimension);
                        if (removed) return;
                        auto &names = column.Cast<ColumnRefExpression>().ColumnNames();
                        other_source_dimension |= names.size() == 1 ||
                            (names.size() == 2 && (names[0] == Identifier(source_alias) || names[0] == Identifier("_inner")));
                    });
                if (!other_source_dimension) atom_other = false;
            }
            overridden |= atom_overridden;
            other |= atom_other;
        }
        if (overridden && other) {
            throw ParserException("AT dimension replacement cannot separate a mixed VISIBLE predicate; "
                                  "use an explicit AT WHERE predicate");
        }
        if (overridden) continue;
        if (clauses[i].size() == 1) remaining.push_back(std::move(clauses[i][0]));
        else remaining.push_back(make_uniq<ConjunctionExpression>(ExpressionType::CONJUNCTION_OR, std::move(clauses[i])));
    }
    if (remaining.empty()) expression = ConstantExpression::FromValue(Value::BOOLEAN(true));
    else if (remaining.size() == 1) expression = std::move(remaining[0]);
    else expression = make_uniq<ConjunctionExpression>(ExpressionType::CONJUNCTION_AND, std::move(remaining));
}
#endif

#if YARDSTICK_GRAMMAR_EXTENSION
void duckdb::RemoveNativeContextDimensions(unique_ptr<ParsedExpression> &expression,
                                           const vector<unique_ptr<ParsedExpression>> &dimensions,
                                           const string &source_alias) {
    vector<unique_ptr<ParsedExpression>> canonical;
    for (auto &dimension : dimensions) {
        canonical.push_back(ContextDimensionReference(*dimension, source_alias));
    }
    DropContextDimensions(expression, canonical, source_alias);
}
#endif

extern "C" char* yardstick_rewrite_visible_filter(const char* expression, const char* local_alias,
                                                  const char* const* dimension_names,
                                                  const char* const* dimension_expressions,
                                                  size_t dimension_count,
                                                  const char* const* excluded_dimensions,
                                                  size_t excluded_count, char** error) {
    if (error) *error = nullptr;
#if YARDSTICK_GRAMMAR_EXTENSION
    auto *options = CurrentNativeYardstickParserOptions();
    if (!options || !expression) return nullptr;
    try {
        auto expressions = YardstickParseExpressionList(expression, *options);
        if (expressions.size() != 1) throw ParserException("Expected one expression in visible filter");
        string visible_source;
        bool joined_visibility = false;
        if (expressions[0]->GetExpressionClass() == ExpressionClass::FUNCTION) {
            auto &function = expressions[0]->Cast<FunctionExpression>();
            auto arguments = YsArgs(function);
            if (YsFuncName(function) == "__yardstick_visible_join" && arguments.size() == 2 &&
                arguments[0]->GetExpressionClass() == ExpressionClass::CONSTANT) {
                visible_source = arguments[0]->Cast<ConstantExpression>().GetLiteral().text;
                expressions[0] = arguments[1]->Copy();
                joined_visibility = true;
            }
        }
        VisibleFilterRewriter rewriter(local_alias ? local_alias : "", *options);
        for (size_t i = 0; i < dimension_count; i++) rewriter.AddDimension(dimension_names[i], dimension_expressions[i]);
        rewriter.Expression(expressions[0], {});
        vector<unique_ptr<ParsedExpression>> excluded;
        for (size_t i = 0; i < excluded_count; i++) {
            auto dimensions = YardstickParseExpressionList(excluded_dimensions[i], *options);
            if (dimensions.size() != 1) throw ParserException("Expected one context dimension");
            excluded.push_back(ContextDimensionReference(*dimensions[0], local_alias ? local_alias : ""));
            rewriter.Expression(dimensions[0], {});
            excluded.push_back(std::move(dimensions[0]));
        }
        if (joined_visibility) {
            if (expressions[0]->GetExpressionClass() != ExpressionClass::SUBQUERY ||
                expressions[0]->Cast<SubqueryExpression>().Subquery()->node->type != QueryNodeType::SELECT_NODE) {
                throw ParserException("Invalid joined visibility context");
            }
            auto &select = expressions[0]->Cast<SubqueryExpression>().SubqueryMutable()->node->Cast<SelectNode>();
            DropContextDimensions(select.where_clause, excluded, visible_source);
            std::function<void(TableRef &)> prune_joins = [&](TableRef &table) {
                if (table.type != TableReferenceType::JOIN) return;
                auto &join = table.Cast<JoinRef>();
                DropContextDimensions(join.condition, excluded, visible_source);
                bool removed_using_dimension = false;
                for (idx_t i = 0; i < join.using_columns.size();) {
                    auto column = make_uniq<ColumnRefExpression>(join.using_columns[i]);
                    auto canonical = ContextDimensionReference(*column, visible_source);
                    bool removed = false;
                    for (auto &dimension : excluded) removed |= canonical->Equals(*dimension);
                    if (removed) {
                        join.using_columns.erase(join.using_columns.begin() + i);
                        removed_using_dimension = true;
                    } else i++;
                }
                if (removed_using_dimension && join.using_columns.empty() && !join.condition) {
                    join.condition = ConstantExpression::FromValue(Value::BOOLEAN(true));
                }
                prune_joins(*join.left);
                prune_joins(*join.right);
            };
            if (select.from_table) prune_joins(*select.from_table);
        } else {
            DropContextDimensions(expressions[0], excluded);
        }
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

extern "C" char* yardstick_qualify_expression(const char* expr_str, const char* qualifier,
                                             const char* dimension) {
    if (!expr_str || !qualifier) return nullptr;

    try {
        if (dimension) {
#if YARDSTICK_GRAMMAR_EXTENSION
            if (!CurrentNativeYardstickParserOptions()) return nullptr;
            auto options = YardstickParserOptions();
            auto dimensions = YardstickParseExpressionList(dimension, options);
            if (dimensions.size() != 1 || dimensions[0]->GetExpressionClass() != ExpressionClass::COLUMN_REF) {
                return nullptr;
            }
            auto column = dimensions[0]->Cast<ColumnRefExpression>().GetColumnName();
            auto qualified = YardstickParseExpressionList(string(qualifier) + "." + dimension, options);
            if (qualified.size() != 1 || qualified[0]->GetExpressionClass() != ExpressionClass::COLUMN_REF) {
                return nullptr;
            }
            auto qualified_names = qualified[0]->Cast<ColumnRefExpression>().ColumnNames();
            Parser parser(options);
            parser.ParseQuery(string("SELECT ") + expr_str + " FROM " + qualifier);
            if (parser.statements.size() != 1 || parser.statements[0]->type != StatementType::SELECT_STATEMENT) {
                return nullptr;
            }
            auto &query = parser.statements[0]->Cast<SelectStatement>().node;
            if (query->type != QueryNodeType::SELECT_NODE) return nullptr;
            auto &select = query->Cast<SelectNode>();
            if (select.select_list.size() != 1) return nullptr;
            NativeSubqueryDimensions(*select.select_list[0], select, [&](ColumnRefExpression &reference) {
                if (!reference.IsQualified() && reference.GetColumnName() == column) {
                    reference.ColumnNamesMutable() = qualified_names;
                }
            });
            return safe_strdup(select.select_list[0]->ToString());
#else
            return nullptr;
#endif
        }
        auto expressions = YardstickParseExpressionList(expr_str, YardstickParserOptions());
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
