#pragma once

#include "duckdb.hpp"
#include "yardstick_compat.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/parser/parser_extension.hpp"
#include "duckdb/parser/statement/extension_statement.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/operator/logical_extension_operator.hpp"
#include "duckdb/planner/operator_extension.hpp"

namespace duckdb {

// Forward declarations
BoundStatement yardstick_bind(ClientContext &context, Binder &binder,
                               OperatorExtensionInfo *info, SQLStatement &statement);

// DuckDB main changed parse_function_t to receive the post-PEG-failure token
// tail (vector<SimpleToken>) instead of the raw query string; DuckDB 1.5 and
// earlier pass the query string. CMake detects the callback signature directly.
// Yardstick performs all of its rewriting in yardstick_parser_override (which
// still receives the full query string on both APIs), so on the new signature
// parse_function is a no-op fallback.
#if YARDSTICK_TOKEN_PARSE_FN
ParserExtensionParseResult yardstick_parse(ParserExtensionInfo *,
                                            const vector<SimpleToken> &tokens);
#else
ParserExtensionParseResult yardstick_parse(ParserExtensionInfo *,
                                            const std::string &query);
#endif

ParserExtensionPlanResult yardstick_plan(ParserExtensionInfo *, ClientContext &,
                                          unique_ptr<ParserExtensionParseData>);

ParserOverrideResult yardstick_parser_override(ParserExtensionInfo *info,
                                                const std::string &query,
                                                ParserOptions &options);

// Operator extension: handles binding after parsing
#if YARDSTICK_GRAMMAR_EXTENSION
struct YardstickOperatorInfo : public OperatorExtensionInfo {
    explicit YardstickOperatorInfo(shared_ptr<ParserExtensionInfo> parser_info)
        : parser_info(std::move(parser_info)) {}
    shared_ptr<ParserExtensionInfo> parser_info;
};
#endif

struct YardstickOperatorExtension : public OperatorExtension {
    explicit YardstickOperatorExtension(shared_ptr<ParserExtensionInfo> parser_info = nullptr) : OperatorExtension() {
        Bind = yardstick_bind;
#if YARDSTICK_GRAMMAR_EXTENSION
        operator_info = make_shared_ptr<YardstickOperatorInfo>(std::move(parser_info));
#endif
    }
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

#if YARDSTICK_GRAMMAR_EXTENSION
struct YardstickTemporaryBatch {
    vector<string> view_names;
};

// Column-list declarations and temporary metadata use bind-time lifecycle handling.
struct YardstickDeferredParseData : ParserExtensionParseData {
    string sql;
    ParserOptions options;
    shared_ptr<YardstickTemporaryBatch> temporary_batch;
    bool completes_temporary_batch = false;

    YardstickDeferredParseData(string sql, ParserOptions options,
                              shared_ptr<YardstickTemporaryBatch> temporary_batch = nullptr)
        : sql(std::move(sql)), options(std::move(options)), temporary_batch(std::move(temporary_batch)) {}
    unique_ptr<ParserExtensionParseData> Copy() const override {
        auto result = make_uniq<YardstickDeferredParseData>(sql, options, temporary_batch);
        result->completes_temporary_batch = completes_temporary_batch;
        return std::move(result);
    }
    string ToString() const override { return sql; }
};
#endif

// State stored in ClientContext between parse and bind
class YardstickState : public ClientContextState {
public:
    explicit YardstickState(unique_ptr<ParserExtensionParseData> parse_data)
        : parse_data(std::move(parse_data)) {}
    void QueryEnd() override { parse_data.reset(); }
    unique_ptr<ParserExtensionParseData> parse_data;
};

} // namespace duckdb
