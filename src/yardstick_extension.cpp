#define DUCKDB_EXTENSION_MAIN

#include "yardstick_extension.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/parser/statement/extension_statement.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/connection.hpp"

// Rust FFI - Julian Hyde "Measures in SQL" (arXiv:2406.00251)
extern "C" {
    struct YardstickCreateViewResult {
        bool is_measure_view;
        char *view_name;
        char *clean_sql;
        char *error;
    };

    struct YardstickAggregateResult {
        bool had_aggregate;
        char *expanded_sql;
        char *error;
    };

    bool yardstick_has_as_measure(const char *sql);
    bool yardstick_has_aggregate(const char *sql);
    YardstickCreateViewResult yardstick_process_create_view(const char *sql);
    YardstickAggregateResult yardstick_expand_aggregate(const char *sql);
    void yardstick_free(char *ptr);
    void yardstick_free_create_view_result(YardstickCreateViewResult result);
    void yardstick_free_aggregate_result(YardstickAggregateResult result);
}

namespace duckdb {

//=============================================================================
// TABLE FUNCTION: yardstick(sql) - Execute SQL with AGGREGATE() expansion
//=============================================================================

struct YardstickQueryData : public TableFunctionData {
    string original_sql;
    string rewritten_sql;
    unique_ptr<QueryResult> result;
    bool done = false;
};

static unique_ptr<FunctionData> YardstickQueryBind(ClientContext &context,
                                                     TableFunctionBindInput &input,
                                                     vector<LogicalType> &return_types,
                                                     vector<string> &names) {
    auto data = make_uniq<YardstickQueryData>();
    data->original_sql = input.inputs[0].GetValue<string>();

    // Rewrite the SQL using our Rust code
    if (yardstick_has_aggregate(data->original_sql.c_str())) {
        YardstickAggregateResult result = yardstick_expand_aggregate(data->original_sql.c_str());
        if (result.error) {
            string error_msg(result.error);
            yardstick_free_aggregate_result(result);
            throw InvalidInputException("Failed to expand AGGREGATE: %s", error_msg);
        }
        if (result.had_aggregate) {
            data->rewritten_sql = string(result.expanded_sql);
        } else {
            data->rewritten_sql = data->original_sql;
        }
        yardstick_free_aggregate_result(result);
    } else {
        data->rewritten_sql = data->original_sql;
    }

    // Execute the rewritten query to get schema
    Connection con(*context.db);
    auto query_result = con.Query(data->rewritten_sql);

    if (query_result->HasError()) {
        throw InvalidInputException("Query error: %s", query_result->GetError());
    }

    // Extract return types and names from result
    for (idx_t i = 0; i < query_result->ColumnCount(); i++) {
        return_types.push_back(query_result->types[i]);
        names.push_back(query_result->names[i]);
    }

    // Store the result for iteration
    data->result = std::move(query_result);

    return std::move(data);
}

static void YardstickQueryFunction(ClientContext &context, TableFunctionInput &data_p,
                                    DataChunk &output) {
    auto &data = data_p.bind_data->CastNoConst<YardstickQueryData>();

    if (data.done || !data.result) {
        output.SetCardinality(0);
        return;
    }

    // Fetch next chunk from result
    auto chunk = data.result->Fetch();
    if (!chunk || chunk->size() == 0) {
        data.done = true;
        output.SetCardinality(0);
        return;
    }

    // Copy data to output
    output.SetCardinality(chunk->size());
    for (idx_t col = 0; col < chunk->ColumnCount(); col++) {
        output.data[col].Reference(chunk->data[col]);
    }
}

//=============================================================================
// PARSER EXTENSION
//=============================================================================

// Check if query starts with SEMANTIC keyword (case insensitive)
static bool StartsWithSemantic(const std::string &query, std::string &stripped_query) {
    size_t start = 0;
    while (start < query.size() && std::isspace(query[start])) {
        start++;
    }

    const char *prefix = "SEMANTIC";
    size_t prefix_len = 8;

    if (query.size() - start < prefix_len) {
        return false;
    }

    for (size_t i = 0; i < prefix_len; i++) {
        if (std::toupper(query[start + i]) != prefix[i]) {
            return false;
        }
    }

    if (start + prefix_len < query.size() && !std::isspace(query[start + prefix_len])) {
        return false;
    }

    stripped_query = query.substr(start + prefix_len);
    return true;
}

ParserExtensionParseResult yardstick_parse(ParserExtensionInfo *,
                                            const std::string &query) {

    // Determine the SQL to check (strip SEMANTIC prefix if present)
    std::string sql_to_check = query;
    std::string semantic_stripped;
    bool had_semantic_prefix = StartsWithSemantic(query, semantic_stripped);
    if (had_semantic_prefix) {
        sql_to_check = semantic_stripped;
    }

    // Check for AGGREGATE() function
    if (yardstick_has_aggregate(sql_to_check.c_str())) {
        YardstickAggregateResult result = yardstick_expand_aggregate(sql_to_check.c_str());

        if (result.error) {
            string error_msg(result.error);
            yardstick_free_aggregate_result(result);
            return ParserExtensionParseResult(error_msg);
        }

        if (result.had_aggregate) {
            string expanded_sql(result.expanded_sql);
            yardstick_free_aggregate_result(result);

            // Escape single quotes for embedding in string literal
            string escaped_sql;
            for (char c : expanded_sql) {
                if (c == '\'') {
                    escaped_sql += "''";
                } else {
                    escaped_sql += c;
                }
            }

            // Wrap in table function call
            string wrapper_sql = "SELECT * FROM yardstick('" + escaped_sql + "')";

            Parser parser;
            parser.ParseQuery(wrapper_sql);
            auto statements = std::move(parser.statements);

            if (statements.empty()) {
                return ParserExtensionParseResult("Table function wrapper produced no statements");
            }

            return ParserExtensionParseResult(
                make_uniq_base<ParserExtensionParseData, YardstickParseData>(
                    std::move(statements[0])));
        }

        yardstick_free_aggregate_result(result);
    }

    // Check for CREATE VIEW with AS MEASURE
    if (yardstick_has_as_measure(sql_to_check.c_str())) {
        YardstickCreateViewResult result = yardstick_process_create_view(query.c_str());

        if (result.error) {
            string error_msg(result.error);
            yardstick_free_create_view_result(result);
            return ParserExtensionParseResult(error_msg);
        }

        if (result.is_measure_view) {
            string clean_sql(result.clean_sql);
            yardstick_free_create_view_result(result);

            Parser parser;
            parser.ParseQuery(clean_sql);
            auto statements = std::move(parser.statements);

            if (statements.empty()) {
                return ParserExtensionParseResult("CREATE VIEW produced no statements");
            }

            return ParserExtensionParseResult(
                make_uniq_base<ParserExtensionParseData, YardstickParseData>(
                    std::move(statements[0])));
        }

        yardstick_free_create_view_result(result);
    }

    // Not a yardstick query, let DuckDB handle it
    return ParserExtensionParseResult();
}

ParserExtensionPlanResult yardstick_plan(ParserExtensionInfo *,
                                          ClientContext &context,
                                          unique_ptr<ParserExtensionParseData> parse_data) {
    auto state = make_shared_ptr<YardstickState>(std::move(parse_data));
    context.registered_state->Remove("yardstick");
    context.registered_state->Insert("yardstick", state);
    throw BinderException("Use yardstick_bind instead");
}

BoundStatement yardstick_bind(ClientContext &context, Binder &binder,
                               OperatorExtensionInfo *info, SQLStatement &statement) {
    switch (statement.type) {
    case StatementType::EXTENSION_STATEMENT: {
        auto &ext_statement = dynamic_cast<ExtensionStatement &>(statement);

        if (ext_statement.extension.parse_function == yardstick_parse) {
            auto lookup = context.registered_state->Get<YardstickState>("yardstick");
            if (lookup) {
                auto state = (YardstickState *)lookup.get();
                auto parse_data = dynamic_cast<YardstickParseData *>(state->parse_data.get());

                shared_ptr<Binder> yardstick_binder;
                if (parse_data->statement->type == StatementType::SELECT_STATEMENT) {
                    yardstick_binder = Binder::CreateBinder(context);
                } else {
                    yardstick_binder = Binder::CreateBinder(context, &binder);
                }

                return yardstick_binder->Bind(*(parse_data->statement));
            }
            throw BinderException("Registered state not found");
        }
    }
    default:
        return {};
    }
}

//=============================================================================
// EXTENSION LOADING
//=============================================================================

static void LoadInternal(ExtensionLoader &loader) {
    auto &db = loader.GetDatabaseInstance();
    auto &config = DBConfig::GetConfig(db);

    // Register parser extension
    YardstickParserExtension parser;
    config.parser_extensions.push_back(parser);

    // Register operator extension
    config.operator_extensions.push_back(make_uniq<YardstickOperatorExtension>());

    // Register table function for AGGREGATE() expansion
    TableFunction query_func("yardstick", {LogicalType::VARCHAR},
                             YardstickQueryFunction, YardstickQueryBind);
    loader.RegisterFunction(query_func);
}

void YardstickExtension::Load(ExtensionLoader &loader) {
    LoadInternal(loader);
}

std::string YardstickExtension::Version() const {
#ifdef EXT_VERSION_YARDSTICK
    return EXT_VERSION_YARDSTICK;
#else
    return "0.1.0";
#endif
}

} // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(yardstick, loader) {
    duckdb::LoadInternal(loader);
}

}
