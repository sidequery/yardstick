#pragma once

#include "duckdb.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/parser/statement/extension_statement.hpp"

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
struct YardstickParserExtension : public ParserExtension {
    YardstickParserExtension() : ParserExtension() {
        parse_function = yardstick_parse;
        plan_function = yardstick_plan;
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
