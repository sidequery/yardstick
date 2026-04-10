#pragma once

#include "duckdb.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/parser/parser_extension.hpp"
#include "duckdb/parser/statement/extension_statement.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/operator/logical_extension_operator.hpp"
#include "duckdb/planner/operator_extension.hpp"

namespace duckdb {

// Main extension class
class YardstickExtension : public Extension {
public:
    void Load(ExtensionLoader &loader) override;
    std::string Name() override { return "yardstick"; }
    std::string Version() const override;
};

// Forward declarations
BoundStatement yardstick_bind(ClientContext &context, Binder &binder,
                               OperatorExtensionInfo *info, SQLStatement &statement);

ParserExtensionParseResult yardstick_parse(ParserExtensionInfo *,
                                            const std::string &query);

ParserExtensionPlanResult yardstick_plan(ParserExtensionInfo *, ClientContext &,
                                          unique_ptr<ParserExtensionParseData>);

ParserOverrideResult yardstick_parser_override(ParserExtensionInfo *info,
                                                const std::string &query,
                                                ParserOptions &options);

// Operator extension: handles binding after parsing
struct YardstickOperatorExtension : public OperatorExtension {
    YardstickOperatorExtension() : OperatorExtension() { Bind = yardstick_bind; }
    std::string GetName() override { return "yardstick"; }
    unique_ptr<LogicalExtensionOperator>
    Deserialize(Deserializer &deserializer) override {
        throw InternalException("yardstick operator should not be serialized");
    }
};

// Parser extension: intercepts query strings
// parser_override runs BEFORE DuckDB's native parser, handling all statement types.
// parse_function/plan_function are kept as fallback for when the native parser fails
// (e.g., AT(...) syntax that is not valid SQL).
struct YardstickParserExtension : public ParserExtension {
    YardstickParserExtension() : ParserExtension() {
        parse_function = yardstick_parse;
        plan_function = yardstick_plan;
        parser_override = yardstick_parser_override;
    }
};

// Container for parsed statement (passed between parse and bind phases)
struct YardstickParseData : ParserExtensionParseData {
    unique_ptr<SQLStatement> statement;

    unique_ptr<ParserExtensionParseData> Copy() const override {
        return make_uniq_base<ParserExtensionParseData, YardstickParseData>(
            statement->Copy());
    }
    string ToString() const override { return "YardstickParseData"; }
    YardstickParseData(unique_ptr<SQLStatement> statement)
        : statement(std::move(statement)) {}
};

// State stored in ClientContext between parse and bind
class YardstickState : public ClientContextState {
public:
    explicit YardstickState(unique_ptr<ParserExtensionParseData> parse_data)
        : parse_data(std::move(parse_data)) {}
    void QueryEnd() override { parse_data.reset(); }
    unique_ptr<ParserExtensionParseData> parse_data;
};

} // namespace duckdb
