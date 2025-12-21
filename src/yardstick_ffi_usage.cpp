/**
 * yardstick_ffi_usage.cpp - Example usage of C struct-based FFI
 *
 * This file demonstrates how to use the new FFI functions from C++.
 * It can be used as a reference for integrating with yardstick_extension.cpp.
 */

#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>

// Include the FFI header
#include "yardstick_ffi.h"

namespace duckdb {
namespace yardstick_ffi {

/**
 * Expand all AGGREGATE() calls in SQL using the new FFI.
 *
 * This function:
 * 1. Finds all AGGREGATE() calls with their AT modifiers
 * 2. Parses the SELECT statement for context (tables, GROUP BY, WHERE)
 * 3. Expands each AGGREGATE() call to proper SQL
 * 4. Applies replacements to produce final SQL
 *
 * @param sql Original SQL string containing AGGREGATE() calls
 * @return Expanded SQL with AGGREGATE() calls replaced
 * @throws std::runtime_error on parse or expansion errors
 */
std::string ExpandAggregatesV2(const std::string& sql) {
    // Step 1: Find all AGGREGATE() calls
    YardstickAggregateCallList* agg_list = yardstick_find_aggregates(sql.c_str());

    if (agg_list == nullptr) {
        throw std::runtime_error("Failed to allocate aggregate list");
    }

    if (agg_list->error != nullptr) {
        std::string error(agg_list->error);
        yardstick_free_aggregate_list(agg_list);
        throw std::runtime_error("Error finding aggregates: " + error);
    }

    if (agg_list->count == 0) {
        yardstick_free_aggregate_list(agg_list);
        return sql;  // No AGGREGATE() calls found
    }

    // Step 2: Parse SELECT for context
    YardstickSelectInfo* select_info = yardstick_parse_select(sql.c_str());

    if (select_info == nullptr) {
        yardstick_free_aggregate_list(agg_list);
        throw std::runtime_error("Failed to allocate select info");
    }

    if (select_info->error != nullptr) {
        std::string error(select_info->error);
        yardstick_free_select_info(select_info);
        yardstick_free_aggregate_list(agg_list);
        throw std::runtime_error("Error parsing SELECT: " + error);
    }

    // Step 3: Build replacements for each AGGREGATE() call
    // Store as (start, end, replacement)
    std::vector<std::tuple<uint32_t, uint32_t, std::string>> replacements;
    replacements.reserve(agg_list->count);

    for (size_t i = 0; i < agg_list->count; i++) {
        const YardstickAggregateCall& call = agg_list->calls[i];

        // Look up measure aggregation function
        // In real usage, this would query a measure catalog
        // For now, default to "SUM"
        const char* agg_func = "SUM";

        // TODO: Look up from measure registry:
        // YardstickMeasureAggResult measure_info =
        //     yardstick_get_measure_aggregation(call.measure_name);
        // if (measure_info.aggregation) {
        //     agg_func = measure_info.aggregation;
        // }

        // Expand the AGGREGATE() call using context
        char* expanded = yardstick_expand_aggregate_call(
            call.measure_name,
            agg_func,
            call.modifiers,
            call.modifier_count,
            select_info->primary_table,
            nullptr,  // outer_alias - could pass table alias here
            select_info->where_clause,
            select_info->group_by_cols,
            select_info->group_by_count
        );

        if (expanded != nullptr) {
            replacements.emplace_back(
                call.start_pos,
                call.end_pos,
                std::string(expanded)
            );
            yardstick_free_string(expanded);
        }
    }

    // Sort replacements by position (descending) for stable string replacement
    std::sort(replacements.begin(), replacements.end(),
        [](const auto& a, const auto& b) {
            return std::get<0>(a) > std::get<0>(b);
        });

    // Step 4: Apply replacements from end to beginning
    std::string result = sql;
    for (const auto& [start, end, repl] : replacements) {
        if (start <= result.size() && end <= result.size()) {
            result = result.substr(0, start) + repl + result.substr(end);
        }
    }

    // Cleanup
    yardstick_free_select_info(select_info);
    yardstick_free_aggregate_list(agg_list);

    return result;
}

/**
 * Check if SQL contains any AGGREGATE() calls.
 * Fast path check before doing full parsing.
 */
bool HasAggregateCalls(const std::string& sql) {
    return yardstick_has_aggregate(sql.c_str());
}

/**
 * Parse a measure expression to extract aggregation info.
 *
 * @param expr Expression string, e.g., "SUM(amount)"
 * @return Tuple of (is_aggregate, agg_func, inner_expr)
 */
std::tuple<bool, std::string, std::string> ParseMeasureExpression(const std::string& expr) {
    YardstickExpressionInfo* info = yardstick_parse_expression(expr.c_str());

    if (info == nullptr || info->error != nullptr) {
        if (info != nullptr) {
            yardstick_free_expression_info(info);
        }
        return {false, "", ""};
    }

    bool is_aggregate = info->is_aggregate;
    std::string agg_func = info->aggregate_func ? info->aggregate_func : "";
    std::string inner = info->inner_expr ? info->inner_expr : "";

    yardstick_free_expression_info(info);
    return {is_aggregate, agg_func, inner};
}

/**
 * Get information about a SELECT statement's structure.
 */
struct SelectStatementInfo {
    std::vector<std::string> select_items;
    std::vector<std::string> tables;
    std::string primary_table;
    std::vector<std::string> group_by_cols;
    std::string where_clause;
    bool has_group_by;
    bool group_by_all;
};

SelectStatementInfo GetSelectInfo(const std::string& sql) {
    SelectStatementInfo result;

    YardstickSelectInfo* info = yardstick_parse_select(sql.c_str());
    if (info == nullptr || info->error != nullptr) {
        if (info != nullptr) {
            yardstick_free_select_info(info);
        }
        return result;
    }

    // Extract select items
    for (size_t i = 0; i < info->item_count; i++) {
        if (info->items[i].expression_sql) {
            result.select_items.push_back(info->items[i].expression_sql);
        }
    }

    // Extract tables
    for (size_t i = 0; i < info->table_count; i++) {
        if (info->tables[i].table_name) {
            result.tables.push_back(info->tables[i].table_name);
        }
    }

    // Primary table
    if (info->primary_table) {
        result.primary_table = info->primary_table;
    }

    // Group by columns
    for (size_t i = 0; i < info->group_by_count; i++) {
        if (info->group_by_cols[i]) {
            result.group_by_cols.push_back(info->group_by_cols[i]);
        }
    }

    // Where clause
    if (info->where_clause) {
        result.where_clause = info->where_clause;
    }

    result.has_group_by = info->has_group_by;
    result.group_by_all = info->group_by_all;

    yardstick_free_select_info(info);
    return result;
}

/**
 * Example: Integration point for parser extension.
 *
 * This shows how yardstick_parse could be refactored to use the new FFI.
 */
// ParserExtensionParseResult yardstick_parse_with_new_ffi(
//     ParserExtensionInfo*,
//     const std::string& query
// ) {
//     // Quick check - no AGGREGATE() means nothing to do
//     if (!HasAggregateCalls(query)) {
//         return ParserExtensionParseResult();  // Not our syntax
//     }
//
//     try {
//         // Use new FFI for expansion
//         std::string expanded = ExpandAggregatesV2(query);
//
//         // Parse the expanded SQL with DuckDB's parser
//         Parser parser;
//         parser.ParseQuery(expanded);
//
//         if (parser.statements.empty()) {
//             return ParserExtensionParseResult("Expansion produced no statements");
//         }
//
//         return ParserExtensionParseResult(
//             make_uniq_base<ParserExtensionParseData, YardstickParseData>(
//                 std::move(parser.statements[0])));
//     } catch (const std::exception& e) {
//         return ParserExtensionParseResult(e.what());
//     }
// }

} // namespace yardstick_ffi
} // namespace duckdb

// Example main for testing (not built in normal extension build)
#ifdef YARDSTICK_FFI_TEST_MAIN
#include <iostream>

int main() {
    using namespace duckdb::yardstick_ffi;

    // Test 1: Simple AGGREGATE()
    {
        std::string sql = "SELECT region, AGGREGATE(revenue) FROM sales GROUP BY region";
        std::cout << "Input:  " << sql << "\n";
        std::cout << "Output: " << ExpandAggregatesV2(sql) << "\n\n";
    }

    // Test 2: AGGREGATE with AT
    {
        std::string sql = "SELECT region, AGGREGATE(revenue) AT (ALL) FROM sales GROUP BY region";
        std::cout << "Input:  " << sql << "\n";
        std::cout << "Output: " << ExpandAggregatesV2(sql) << "\n\n";
    }

    // Test 3: Parse expression
    {
        auto [is_agg, func, inner] = ParseMeasureExpression("SUM(amount)");
        std::cout << "Expression: SUM(amount)\n";
        std::cout << "  is_aggregate: " << (is_agg ? "true" : "false") << "\n";
        std::cout << "  agg_func: " << func << "\n";
        std::cout << "  inner: " << inner << "\n\n";
    }

    // Test 4: Get SELECT info
    {
        std::string sql = "SELECT region, SUM(amount) FROM sales WHERE year = 2024 GROUP BY region";
        auto info = GetSelectInfo(sql);
        std::cout << "SELECT info for: " << sql << "\n";
        std::cout << "  primary_table: " << info.primary_table << "\n";
        std::cout << "  has_group_by: " << (info.has_group_by ? "true" : "false") << "\n";
        std::cout << "  where_clause: " << info.where_clause << "\n";
    }

    return 0;
}
#endif
