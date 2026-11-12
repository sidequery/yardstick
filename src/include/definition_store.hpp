#pragma once

#include "yardstick_compat.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/parser/parser_extension.hpp"
#include "duckdb/parser/statement/extension_statement.hpp"
#include "duckdb/planner/bound_statement.hpp"

namespace duckdb {
class BaseTableRef;
class Binder;
class ClientContext;
class TableRef;
struct DBConfig;

// DuckDB's ExtensionStatement::Copy drops SQLStatement metadata. Prepared
// statements need that metadata to validate parameters before rebinding.
class YardstickDefinitionStatement : public ExtensionStatement {
public:
    YardstickDefinitionStatement(ParserExtension extension, unique_ptr<ParserExtensionParseData> data)
        : ExtensionStatement(std::move(extension), std::move(data)) {}

    unique_ptr<SQLStatement> Copy() const override {
        auto result = make_uniq<YardstickDefinitionStatement>(extension, parse_data->Copy());
        result->query = query;
        result->stmt_location = stmt_location;
        result->named_param_map = named_param_map;
#if YARDSTICK_NEW_EXPR_API
        result->has_anonymous_parameters = has_anonymous_parameters;
#else
        result->stmt_length = stmt_length;
#endif
        return std::move(result);
    }
};

// The parser has no ClientContext. Once a session selects a store, defer its
// statements until binding, where the setting and transaction are available.
bool RouteYardstickDefinitionStatements(const string &query, const ParserOptions &options);
bool YardstickDefinitionParsingBypassed();
bool YardstickDefinitionStoreConfigured(const ClientContext &context);
unique_ptr<TableRef> YardstickDefinitionQueryRef(ClientContext &context, const string &sql,
                                                const ParserOptions &options, ParserExtensionInfo *parser_info);
void HandleAggregateWarnings(ClientContext &context, const string &warnings);

struct YardstickDefinitionParseData : ParserExtensionParseData {
    string sql;
    ParserOptions options;
    bool initialize;

    YardstickDefinitionParseData(string sql, ParserOptions options, bool initialize = false)
        : sql(std::move(sql)), options(std::move(options)), initialize(initialize) {}
    unique_ptr<ParserExtensionParseData> Copy() const override {
        return make_uniq_base<ParserExtensionParseData, YardstickDefinitionParseData>(sql, options, initialize);
    }
    string ToString() const override { return sql; }
};

void RegisterYardstickDefinitionStore(DBConfig &config);
BoundStatement BindYardstickDefinitionStatement(ClientContext &context, Binder &binder,
                                                const YardstickDefinitionParseData &data,
                                                ParserExtensionInfo *parser_info);
// Preserve legacy behavior when another session enabled parser routing but
// this session has no configured store.
BoundStatement BindYardstickUnstoredStatement(ClientContext &context, Binder &binder,
                                             const string &sql, const ParserOptions &options);

// Supply a unique query-local measure key, retaining the SQL table alias.
string YardstickStoredRelationName(const BaseTableRef &ref);

class YardstickTemporaryDefinitionScope {
public:
    virtual ~YardstickTemporaryDefinitionScope() = default;
};
unique_ptr<YardstickTemporaryDefinitionScope> EnterYardstickTemporaryDefinitions(ClientContext &context);
} // namespace duckdb
