#pragma once

#include "duckdb/parser/parsed_expression_iterator.hpp"
#include "duckdb/parser/parsed_data/create_table_info.hpp"
#include "duckdb/parser/parsed_data/create_view_info.hpp"
#include "duckdb/parser/query_node/update_query_node.hpp"
#include "duckdb/parser/statement/copy_statement.hpp"
#include "duckdb/parser/statement/create_statement.hpp"
#include "duckdb/parser/statement/delete_statement.hpp"
#include "duckdb/parser/statement/explain_statement.hpp"
#include "duckdb/parser/statement/insert_statement.hpp"
#include "duckdb/parser/statement/select_statement.hpp"
#include "duckdb/parser/statement/update_statement.hpp"

namespace duckdb {

// The callback owns expression recursion, including scalar subqueries. Return
// false for unsupported wrappers so discovery never accepts a partial result.
inline bool EnumerateNativeStatementExpressions(
    SQLStatement &statement, const std::function<void(unique_ptr<ParsedExpression> &)> &callback) {
    switch (statement.type) {
    case StatementType::SELECT_STATEMENT:
        ParsedExpressionIterator::EnumerateQueryNodeChildren(*statement.Cast<SelectStatement>().node, callback);
        return true;
    case StatementType::INSERT_STATEMENT:
        ParsedExpressionIterator::EnumerateQueryNodeChildren(*statement.Cast<InsertStatement>().node, callback);
        return true;
    case StatementType::UPDATE_STATEMENT:
        ParsedExpressionIterator::EnumerateQueryNodeChildren(*statement.Cast<UpdateStatement>().node, callback);
        return true;
    case StatementType::DELETE_STATEMENT:
        ParsedExpressionIterator::EnumerateQueryNodeChildren(*statement.Cast<DeleteStatement>().node, callback);
        return true;
    case StatementType::EXPLAIN_STATEMENT:
        return EnumerateNativeStatementExpressions(*statement.Cast<ExplainStatement>().stmt, callback);
    case StatementType::CREATE_STATEMENT: {
        auto &info = *statement.Cast<CreateStatement>().info;
        if (info.type == CatalogType::VIEW_ENTRY) {
            auto &view = info.Cast<CreateViewInfo>();
            return view.query && EnumerateNativeStatementExpressions(*view.query, callback);
        }
        if (info.type != CatalogType::TABLE_ENTRY) {
            return false;
        }
        auto &table = info.Cast<CreateTableInfo>();
        // Defaults, generated columns and CHECK constraints have separate AST
        // owners. Keep those forms on compatibility lowering until traversed.
        if (!table.query || !table.columns.empty() || !table.constraints.empty()) {
            return false;
        }
        EnumerateNativeStatementExpressions(*table.query, callback);
        for (auto &expression : table.partition_keys) {
            callback(expression);
        }
        for (auto &expression : table.sort_keys) {
            callback(expression);
        }
        for (auto &option : table.options) {
            if (option.second) {
                callback(option.second);
            }
        }
        return true;
    }
    case StatementType::COPY_STATEMENT: {
        auto &copy = *statement.Cast<CopyStatement>().info;
        if (copy.select_statement) {
            ParsedExpressionIterator::EnumerateQueryNodeChildren(*copy.select_statement, callback);
        }
        if (copy.file_path_expression) {
            callback(copy.file_path_expression);
        }
        for (auto &option : copy.parsed_options) {
            if (option.second) {
                callback(option.second);
            }
        }
        return true;
    }
    default:
        return false;
    }
}

} // namespace duckdb
