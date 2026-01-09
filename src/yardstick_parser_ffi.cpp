/**
 * yardstick_parser_ffi.cpp - C FFI implementation using DuckDB's native parser
 *
 * This file provides FFI functions that use DuckDB's parser to analyze SQL
 * and return results as C structs that can be consumed by Rust.
 *
 * Flow: Rust code -> calls C FFI functions -> C++ uses duckdb::Parser -> returns C structs to Rust
 */

#include "yardstick_ffi.h"

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
#include "duckdb/parser/tableref.hpp"
#include "duckdb/parser/tableref/basetableref.hpp"
#include "duckdb/parser/tableref/joinref.hpp"
#include "duckdb/parser/tableref/subqueryref.hpp"
#include "duckdb/parser/group_by_node.hpp"
#include "duckdb/common/string_util.hpp"

#include <cstring>
#include <vector>
#include <string>

using namespace duckdb;

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
        "array_agg", "list"
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

static void FindAggregateCalls(ParsedExpression* expr, std::vector<AggregateCallInfo>& results);
static void CollectTablesFromTableRef(TableRef* ref, std::vector<YardstickTableRef>& tables);
static bool ExpressionContainsAggregate(ParsedExpression* expr);
static bool ExpressionContainsMeasureRef(ParsedExpression* expr);
static void QualifyColumnRefs(ParsedExpression* expr, const std::string& qualifier);

//=============================================================================
// AST Walking: Find AGGREGATE() function calls
//=============================================================================

static void FindAggregateCalls(ParsedExpression* expr, std::vector<AggregateCallInfo>& results) {
    if (!expr) return;

    switch (expr->expression_class) {
        case ExpressionClass::FUNCTION: {
            auto* func = static_cast<FunctionExpression*>(expr);
            std::string lower_name = StringUtil::Lower(func->function_name);

            if (lower_name == "aggregate") {
                AggregateCallInfo info;

                // Get measure name from first argument
                if (!func->children.empty()) {
                    // First argument should be measure name (column ref or string)
                    auto* first_arg = func->children[0].get();
                    if (first_arg->expression_class == ExpressionClass::COLUMN_REF) {
                        auto* col = static_cast<ColumnRefExpression*>(first_arg);
                        info.measure_name = col->GetColumnName();
                    } else {
                        info.measure_name = first_arg->ToString();
                    }
                }

                // Get position from query_location
                if (expr->query_location.IsValid()) {
                    info.start_pos = static_cast<uint32_t>(expr->query_location.GetIndex());
                } else {
                    info.start_pos = 0;
                }
                // End position is harder without lexer info; estimate from ToString
                info.end_pos = info.start_pos + static_cast<uint32_t>(expr->ToString().length());

                // Parse AT modifiers from remaining arguments
                // AT syntax in DuckDB typically appears as special function arguments
                // For now, we look for patterns like AT(ALL), AT(dimension), etc.
                for (size_t i = 1; i < func->children.size(); i++) {
                    auto* arg = func->children[i].get();

                    // Check if this is an AT modifier call
                    if (arg->expression_class == ExpressionClass::FUNCTION) {
                        auto* at_func = static_cast<FunctionExpression*>(arg);
                        std::string at_name = StringUtil::Lower(at_func->function_name);

                        if (at_name == "at") {
                            YardstickAtModifier mod;
                            mod.dimension = nullptr;
                            mod.value = nullptr;

                            if (!at_func->children.empty()) {
                                auto* at_arg = at_func->children[0].get();
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
                                } else if (at_func->children.size() >= 2) {
                                    // WHERE modifier or ALL dimension
                                    auto* second = at_func->children[1].get();
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
            for (auto& child : func->children) {
                FindAggregateCalls(child.get(), results);
            }
            if (func->filter) {
                FindAggregateCalls(func->filter.get(), results);
            }
            break;
        }

        case ExpressionClass::COMPARISON: {
            auto* comp = static_cast<ComparisonExpression*>(expr);
            FindAggregateCalls(comp->left.get(), results);
            FindAggregateCalls(comp->right.get(), results);
            break;
        }

        case ExpressionClass::CONJUNCTION: {
            auto* conj = static_cast<ConjunctionExpression*>(expr);
            for (auto& child : conj->children) {
                FindAggregateCalls(child.get(), results);
            }
            break;
        }

        case ExpressionClass::OPERATOR: {
            auto* op = static_cast<OperatorExpression*>(expr);
            for (auto& child : op->children) {
                FindAggregateCalls(child.get(), results);
            }
            break;
        }

        case ExpressionClass::CASE: {
            auto* case_expr = static_cast<CaseExpression*>(expr);
            for (auto& check : case_expr->case_checks) {
                FindAggregateCalls(check.when_expr.get(), results);
                FindAggregateCalls(check.then_expr.get(), results);
            }
            if (case_expr->else_expr) {
                FindAggregateCalls(case_expr->else_expr.get(), results);
            }
            break;
        }

        case ExpressionClass::CAST: {
            auto* cast = static_cast<CastExpression*>(expr);
            FindAggregateCalls(cast->child.get(), results);
            break;
        }

        case ExpressionClass::SUBQUERY: {
            auto* subq = static_cast<SubqueryExpression*>(expr);
            if (subq->child) {
                FindAggregateCalls(subq->child.get(), results);
            }
            // Note: We don't recurse into the subquery itself
            break;
        }

        case ExpressionClass::WINDOW: {
            auto* window = static_cast<WindowExpression*>(expr);
            for (auto& child : window->children) {
                FindAggregateCalls(child.get(), results);
            }
            for (auto& part : window->partitions) {
                FindAggregateCalls(part.get(), results);
            }
            if (window->filter_expr) {
                FindAggregateCalls(window->filter_expr.get(), results);
            }
            break;
        }

        case ExpressionClass::BETWEEN: {
            auto* between = static_cast<BetweenExpression*>(expr);
            FindAggregateCalls(between->input.get(), results);
            FindAggregateCalls(between->lower.get(), results);
            FindAggregateCalls(between->upper.get(), results);
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

static void CollectTablesFromTableRef(TableRef* ref, std::vector<YardstickTableRef>& tables) {
    if (!ref) return;

    switch (ref->type) {
        case TableReferenceType::BASE_TABLE: {
            auto* base = static_cast<BaseTableRef*>(ref);
            YardstickTableRef t;
            t.table_name = safe_strdup(base->table_name);
            t.alias = base->alias.empty() ? nullptr : safe_strdup(base->alias);
            t.is_subquery = false;
            tables.push_back(t);
            break;
        }

        case TableReferenceType::JOIN: {
            auto* join = static_cast<JoinRef*>(ref);
            CollectTablesFromTableRef(join->left.get(), tables);
            CollectTablesFromTableRef(join->right.get(), tables);
            break;
        }

        case TableReferenceType::SUBQUERY: {
            auto* subq = static_cast<SubqueryRef*>(ref);
            YardstickTableRef t;
            t.table_name = subq->alias.empty() ? safe_strdup("(subquery)") : safe_strdup(subq->alias);
            t.alias = subq->alias.empty() ? nullptr : safe_strdup(subq->alias);
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

    switch (expr->expression_class) {
        case ExpressionClass::FUNCTION: {
            auto* func = static_cast<FunctionExpression*>(expr);
            if (IsStandardAggregate(func->function_name)) {
                return true;
            }
            for (auto& child : func->children) {
                if (ExpressionContainsAggregate(child.get())) return true;
            }
            if (func->filter && ExpressionContainsAggregate(func->filter.get())) return true;
            return false;
        }

        case ExpressionClass::COMPARISON: {
            auto* comp = static_cast<ComparisonExpression*>(expr);
            return ExpressionContainsAggregate(comp->left.get()) ||
                   ExpressionContainsAggregate(comp->right.get());
        }

        case ExpressionClass::CONJUNCTION: {
            auto* conj = static_cast<ConjunctionExpression*>(expr);
            for (auto& child : conj->children) {
                if (ExpressionContainsAggregate(child.get())) return true;
            }
            return false;
        }

        case ExpressionClass::OPERATOR: {
            auto* op = static_cast<OperatorExpression*>(expr);
            for (auto& child : op->children) {
                if (ExpressionContainsAggregate(child.get())) return true;
            }
            return false;
        }

        case ExpressionClass::CASE: {
            auto* case_expr = static_cast<CaseExpression*>(expr);
            for (auto& check : case_expr->case_checks) {
                if (ExpressionContainsAggregate(check.when_expr.get())) return true;
                if (ExpressionContainsAggregate(check.then_expr.get())) return true;
            }
            return case_expr->else_expr && ExpressionContainsAggregate(case_expr->else_expr.get());
        }

        case ExpressionClass::CAST: {
            auto* cast = static_cast<CastExpression*>(expr);
            return ExpressionContainsAggregate(cast->child.get());
        }

        case ExpressionClass::WINDOW:
            // Window functions are aggregates in a sense
            return true;

        case ExpressionClass::BETWEEN: {
            auto* between = static_cast<BetweenExpression*>(expr);
            return ExpressionContainsAggregate(between->input.get()) ||
                   ExpressionContainsAggregate(between->lower.get()) ||
                   ExpressionContainsAggregate(between->upper.get());
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

    switch (expr->expression_class) {
        case ExpressionClass::FUNCTION: {
            auto* func = static_cast<FunctionExpression*>(expr);
            if (StringUtil::Lower(func->function_name) == "aggregate") {
                return true;
            }
            for (auto& child : func->children) {
                if (ExpressionContainsMeasureRef(child.get())) return true;
            }
            if (func->filter && ExpressionContainsMeasureRef(func->filter.get())) return true;
            return false;
        }

        case ExpressionClass::COMPARISON: {
            auto* comp = static_cast<ComparisonExpression*>(expr);
            return ExpressionContainsMeasureRef(comp->left.get()) ||
                   ExpressionContainsMeasureRef(comp->right.get());
        }

        case ExpressionClass::CONJUNCTION: {
            auto* conj = static_cast<ConjunctionExpression*>(expr);
            for (auto& child : conj->children) {
                if (ExpressionContainsMeasureRef(child.get())) return true;
            }
            return false;
        }

        case ExpressionClass::OPERATOR: {
            auto* op = static_cast<OperatorExpression*>(expr);
            for (auto& child : op->children) {
                if (ExpressionContainsMeasureRef(child.get())) return true;
            }
            return false;
        }

        case ExpressionClass::CASE: {
            auto* case_expr = static_cast<CaseExpression*>(expr);
            for (auto& check : case_expr->case_checks) {
                if (ExpressionContainsMeasureRef(check.when_expr.get())) return true;
                if (ExpressionContainsMeasureRef(check.then_expr.get())) return true;
            }
            return case_expr->else_expr && ExpressionContainsMeasureRef(case_expr->else_expr.get());
        }

        case ExpressionClass::CAST: {
            auto* cast = static_cast<CastExpression*>(expr);
            return ExpressionContainsMeasureRef(cast->child.get());
        }

        case ExpressionClass::BETWEEN: {
            auto* between = static_cast<BetweenExpression*>(expr);
            return ExpressionContainsMeasureRef(between->input.get()) ||
                   ExpressionContainsMeasureRef(between->lower.get()) ||
                   ExpressionContainsMeasureRef(between->upper.get());
        }

        default:
            return false;
    }
}

static void QualifyColumnRefs(ParsedExpression* expr, const std::string& qualifier) {
    if (!expr) return;

    switch (expr->expression_class) {
        case ExpressionClass::COLUMN_REF: {
            auto* col = static_cast<ColumnRefExpression*>(expr);
            if (col->column_names.size() == 1) {
                col->column_names.insert(col->column_names.begin(), qualifier);
            }
            break;
        }
        case ExpressionClass::FUNCTION: {
            auto* func = static_cast<FunctionExpression*>(expr);
            for (auto& child : func->children) {
                QualifyColumnRefs(child.get(), qualifier);
            }
            if (func->filter) {
                QualifyColumnRefs(func->filter.get(), qualifier);
            }
            break;
        }
        case ExpressionClass::COMPARISON: {
            auto* comp = static_cast<ComparisonExpression*>(expr);
            QualifyColumnRefs(comp->left.get(), qualifier);
            QualifyColumnRefs(comp->right.get(), qualifier);
            break;
        }
        case ExpressionClass::CONJUNCTION: {
            auto* conj = static_cast<ConjunctionExpression*>(expr);
            for (auto& child : conj->children) {
                QualifyColumnRefs(child.get(), qualifier);
            }
            break;
        }
        case ExpressionClass::OPERATOR: {
            auto* op = static_cast<OperatorExpression*>(expr);
            for (auto& child : op->children) {
                QualifyColumnRefs(child.get(), qualifier);
            }
            break;
        }
        case ExpressionClass::CASE: {
            auto* case_expr = static_cast<CaseExpression*>(expr);
            for (auto& check : case_expr->case_checks) {
                QualifyColumnRefs(check.when_expr.get(), qualifier);
                QualifyColumnRefs(check.then_expr.get(), qualifier);
            }
            if (case_expr->else_expr) {
                QualifyColumnRefs(case_expr->else_expr.get(), qualifier);
            }
            break;
        }
        case ExpressionClass::CAST: {
            auto* cast = static_cast<CastExpression*>(expr);
            QualifyColumnRefs(cast->child.get(), qualifier);
            break;
        }
        case ExpressionClass::SUBQUERY: {
            auto* subq = static_cast<SubqueryExpression*>(expr);
            if (subq->child) {
                QualifyColumnRefs(subq->child.get(), qualifier);
            }
            break;
        }
        case ExpressionClass::WINDOW: {
            auto* window = static_cast<WindowExpression*>(expr);
            for (auto& child : window->children) {
                QualifyColumnRefs(child.get(), qualifier);
            }
            for (auto& part : window->partitions) {
                QualifyColumnRefs(part.get(), qualifier);
            }
            if (window->filter_expr) {
                QualifyColumnRefs(window->filter_expr.get(), qualifier);
            }
            break;
        }
        case ExpressionClass::BETWEEN: {
            auto* between = static_cast<BetweenExpression*>(expr);
            QualifyColumnRefs(between->input.get(), qualifier);
            QualifyColumnRefs(between->lower.get(), qualifier);
            QualifyColumnRefs(between->upper.get(), qualifier);
            break;
        }
        default:
            break;
    }
}

//=============================================================================
// FFI Implementation: yardstick_find_aggregates
//=============================================================================

extern "C" YardstickAggregateCallList* yardstick_find_aggregates(const char* sql) {
    auto* result = new YardstickAggregateCallList();
    result->calls = nullptr;
    result->count = 0;
    result->error = nullptr;

    if (!sql) {
        result->error = safe_strdup("NULL SQL input");
        return result;
    }

    try {
        Parser parser;
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
                FindAggregateCalls(expr.get(), aggregates);
            }

            // Search in WHERE clause
            if (select_node->where_clause) {
                FindAggregateCalls(select_node->where_clause.get(), aggregates);
            }

            // Search in HAVING clause
            if (select_node->having) {
                FindAggregateCalls(select_node->having.get(), aggregates);
            }

            // Search in GROUP BY expressions
            for (auto& expr : select_node->groups.group_expressions) {
                FindAggregateCalls(expr.get(), aggregates);
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

    if (!sql) {
        result->error = safe_strdup("NULL SQL input");
        return result;
    }

    try {
        Parser parser;
        parser.ParseQuery(sql);

        if (parser.statements.empty()) {
            result->error = safe_strdup("No statements parsed");
            return result;
        }

        auto& stmt = parser.statements[0];
        if (stmt->type != StatementType::SELECT_STATEMENT) {
            result->error = safe_strdup("Not a SELECT statement");
            return result;
        }

        auto* select_stmt = static_cast<SelectStatement*>(stmt.get());
        if (!select_stmt->node || select_stmt->node->type != QueryNodeType::SELECT_NODE) {
            result->error = safe_strdup("Not a simple SELECT node");
            return result;
        }

        auto* select_node = static_cast<SelectNode*>(select_stmt->node.get());

        // Process SELECT list
        std::vector<YardstickSelectItem> items;
        for (auto& expr : select_node->select_list) {
            YardstickSelectItem item;
            item.expression_sql = safe_strdup(expr->ToString());
            item.alias = expr->alias.empty() ? nullptr : safe_strdup(expr->alias);

            if (expr->query_location.IsValid()) {
                item.start_pos = static_cast<uint32_t>(expr->query_location.GetIndex());
            } else {
                item.start_pos = 0;
            }
            item.end_pos = item.start_pos + static_cast<uint32_t>(expr->ToString().length());

            item.is_aggregate = ExpressionContainsAggregate(expr.get());
            item.is_star = expr->expression_class == ExpressionClass::STAR;
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
        if (select_node->from_table) {
            CollectTablesFromTableRef(select_node->from_table.get(), tables);
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

    for (size_t i = 0; i < info->item_count; i++) {
        free(const_cast<char*>(info->items[i].expression_sql));
        free(const_cast<char*>(info->items[i].alias));
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

//=============================================================================
// FFI Implementation: yardstick_parse_expression
//=============================================================================

extern "C" YardstickExpressionInfo* yardstick_parse_expression(const char* expr_str) {
    auto* result = new YardstickExpressionInfo();
    result->sql = nullptr;
    result->aggregate_func = nullptr;
    result->inner_expr = nullptr;
    result->is_aggregate = false;
    result->is_identifier = false;
    result->error = nullptr;

    if (!expr_str) {
        result->error = safe_strdup("NULL expression input");
        return result;
    }

    try {
        auto expressions = Parser::ParseExpressionList(expr_str);

        if (expressions.empty()) {
            result->error = safe_strdup("No expressions parsed");
            return result;
        }

        auto& expr = expressions[0];

        result->sql = safe_strdup(expr->ToString());
        result->is_identifier = expr->expression_class == ExpressionClass::COLUMN_REF;
        result->is_aggregate = ExpressionContainsAggregate(expr.get());

        // If it's a simple aggregate function, extract the function name and inner expr
        if (expr->expression_class == ExpressionClass::FUNCTION) {
            auto* func = static_cast<FunctionExpression*>(expr.get());
            if (IsStandardAggregate(func->function_name)) {
                result->aggregate_func = safe_strdup(StringUtil::Upper(func->function_name));
                if (!func->children.empty()) {
                    result->inner_expr = safe_strdup(func->children[0]->ToString());
                }
            }
        }

    } catch (const std::exception& e) {
        result->error = safe_strdup(e.what());
    }

    return result;
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
    auto* result = new YardstickCreateViewInfo();
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
        Parser parser;
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
    free(const_cast<char*>(info->clean_sql));

    for (size_t i = 0; i < info->measure_count; i++) {
        free(const_cast<char*>(info->measures[i].column_name));
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
        auto expressions = Parser::ParseExpressionList(expr_str);
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
                correlation = std::string(table_name) + "." + col + " = " +
                              std::string(outer_alias) + "." + col;
            } else {
                // Use the column directly for correlation
                correlation = std::string(table_name) + "." + col + " = " + col;
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
