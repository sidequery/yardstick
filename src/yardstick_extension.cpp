#define DUCKDB_EXTENSION_MAIN

#include "yardstick_extension.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/parser/statement/extension_statement.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/function/function_binder.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/aggregate_function_catalog_entry.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include <sstream>

// Rust FFI
extern "C" {
    struct YardstickRewriteResult {
        char *sql;
        char *error;
        bool was_rewritten;
    };

    // Julian Hyde "Measures in SQL" (arXiv:2406.00251)
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

    char *yardstick_load_yaml(const char *yaml);
    char *yardstick_load_file(const char *path);
    void yardstick_clear(void);
    bool yardstick_is_model(const char *table_name);
    char *yardstick_list_models(void);
    YardstickRewriteResult yardstick_rewrite(const char *sql);
    void yardstick_free(char *ptr);
    void yardstick_free_result(YardstickRewriteResult result);
    char *yardstick_define(const char *definition_sql, const char *db_path, bool replace);
    char *yardstick_autoload(const char *db_path);
    char *yardstick_add_definition(const char *definition_sql, const char *db_path, bool is_replace);
    char *yardstick_use(const char *model_name);

    // Julian Hyde "Measures in SQL" functions
    bool yardstick_has_as_measure(const char *sql);
    bool yardstick_has_aggregate(const char *sql);
    YardstickCreateViewResult yardstick_process_create_view(const char *sql);
    YardstickAggregateResult yardstick_expand_aggregate(const char *sql);
    void yardstick_free_create_view_result(YardstickCreateViewResult result);
    void yardstick_free_aggregate_result(YardstickAggregateResult result);

    // Measure lookup for AGGREGATE() function
    struct YardstickMeasureAggResult {
        char *aggregation;  // "sum", "count", etc. or null
        char *view_name;    // view where measure was found, or null
    };
    YardstickMeasureAggResult yardstick_get_measure_aggregation(const char *column_name);
    void yardstick_free_measure_agg_result(YardstickMeasureAggResult result);
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
// TABLE FUNCTION: yardstick_load(yaml)
//=============================================================================

struct YardstickLoadData : public TableFunctionData {
    string yaml_content;
    bool done = false;
};

static unique_ptr<FunctionData> YardstickLoadBind(ClientContext &context,
                                                    TableFunctionBindInput &input,
                                                    vector<LogicalType> &return_types,
                                                    vector<string> &names) {
    auto result = make_uniq<YardstickLoadData>();
    result->yaml_content = input.inputs[0].GetValue<string>();

    return_types.push_back(LogicalType::VARCHAR);
    names.push_back("result");

    return std::move(result);
}

static void YardstickLoadFunction(ClientContext &context, TableFunctionInput &data_p,
                                   DataChunk &output) {
    auto &data = data_p.bind_data->CastNoConst<YardstickLoadData>();
    if (data.done) {
        return;
    }
    data.done = true;

    char *error = yardstick_load_yaml(data.yaml_content.c_str());
    if (error) {
        string error_msg(error);
        yardstick_free(error);
        throw InvalidInputException("Failed to load semantic models: %s", error_msg);
    }

    output.SetCardinality(1);
    output.SetValue(0, 0, Value("Models loaded successfully"));
}

//=============================================================================
// TABLE FUNCTION: yardstick_load_file(path)
//=============================================================================

struct YardstickLoadFileData : public TableFunctionData {
    string file_path;
    bool done = false;
};

static unique_ptr<FunctionData> YardstickLoadFileBind(ClientContext &context,
                                                        TableFunctionBindInput &input,
                                                        vector<LogicalType> &return_types,
                                                        vector<string> &names) {
    auto result = make_uniq<YardstickLoadFileData>();
    result->file_path = input.inputs[0].GetValue<string>();

    return_types.push_back(LogicalType::VARCHAR);
    names.push_back("result");

    return std::move(result);
}

static void YardstickLoadFileFunction(ClientContext &context, TableFunctionInput &data_p,
                                        DataChunk &output) {
    auto &data = data_p.bind_data->CastNoConst<YardstickLoadFileData>();
    if (data.done) {
        return;
    }
    data.done = true;

    char *error = yardstick_load_file(data.file_path.c_str());
    if (error) {
        string error_msg(error);
        yardstick_free(error);
        throw InvalidInputException("Failed to load semantic models: %s", error_msg);
    }

    output.SetCardinality(1);
    output.SetValue(0, 0, Value("Models loaded from: " + data.file_path));
}

//=============================================================================
// TABLE FUNCTION: yardstick_models()
//=============================================================================

struct YardstickModelsData : public TableFunctionData {
    bool done = false;
};

static unique_ptr<FunctionData> YardstickModelsBind(ClientContext &context,
                                                      TableFunctionBindInput &input,
                                                      vector<LogicalType> &return_types,
                                                      vector<string> &names) {
    return_types.push_back(LogicalType::VARCHAR);
    names.push_back("model_name");
    return make_uniq<YardstickModelsData>();
}

static void YardstickModelsFunction(ClientContext &context, TableFunctionInput &data_p,
                                     DataChunk &output) {
    auto &data = data_p.bind_data->CastNoConst<YardstickModelsData>();
    if (data.done) {
        return;
    }
    data.done = true;

    char *models_str = yardstick_list_models();
    if (!models_str) {
        output.SetCardinality(0);
        return;
    }

    string models(models_str);
    yardstick_free(models_str);

    if (models.empty()) {
        output.SetCardinality(0);
        return;
    }

    // Split by comma
    vector<string> model_names;
    size_t pos = 0;
    while ((pos = models.find(',')) != string::npos) {
        model_names.push_back(models.substr(0, pos));
        models.erase(0, pos + 1);
    }
    if (!models.empty()) {
        model_names.push_back(models);
    }

    output.SetCardinality(model_names.size());
    for (idx_t i = 0; i < model_names.size(); i++) {
        output.SetValue(0, i, Value(model_names[i]));
    }
}

//=============================================================================
// SCALAR FUNCTION: yardstick_rewrite_sql(sql)
//=============================================================================

static void YardstickRewriteSqlFunction(DataChunk &args, ExpressionState &state,
                                          Vector &result) {
    auto &sql_vector = args.data[0];
    UnaryExecutor::Execute<string_t, string_t>(
        sql_vector, result, args.size(), [&](string_t sql) {
            YardstickRewriteResult res = yardstick_rewrite(sql.GetString().c_str());

            if (res.error) {
                string error_msg(res.error);
                yardstick_free_result(res);
                throw InvalidInputException("Rewrite failed: %s", error_msg);
            }

            string rewritten(res.sql);
            yardstick_free_result(res);
            return StringVector::AddString(result, rewritten);
        });
}

//=============================================================================
// SCALAR FUNCTION: aggregate() - Julian Hyde Measures in SQL
//=============================================================================

// Bind data for AGGREGATE function - stores the aggregation function name
struct AggregateBindData : public FunctionData {
    string agg_function;  // "sum", "count", "avg", etc.
    string column_name;   // Original column name
    LogicalType return_type;

    AggregateBindData(string agg_fn, string col, LogicalType ret_type)
        : agg_function(std::move(agg_fn)), column_name(std::move(col)), return_type(std::move(ret_type)) {}

    unique_ptr<FunctionData> Copy() const override {
        return make_uniq<AggregateBindData>(agg_function, column_name, return_type);
    }

    bool Equals(const FunctionData &other_p) const override {
        auto &other = other_p.Cast<AggregateBindData>();
        return agg_function == other.agg_function && column_name == other.column_name;
    }
};

// Bind callback - looks up measure definition and stores aggregation function
static unique_ptr<FunctionData> AggregateBindFunction(ClientContext &context,
                                                        ScalarFunction &bound_function,
                                                        vector<unique_ptr<Expression>> &arguments) {
    if (arguments.empty()) {
        throw BinderException("AGGREGATE() requires at least one argument");
    }

    // Get the column name from the first argument
    string column_name;
    auto &arg = arguments[0];

    // Try to extract column name from the expression
    if (arg->GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
        auto &colref = arg->Cast<BoundColumnRefExpression>();
        column_name = colref.GetName();
    } else {
        // For other expression types, try to use ToString
        column_name = arg->ToString();
    }

    // Look up the measure aggregation function via Rust
    YardstickMeasureAggResult result = yardstick_get_measure_aggregation(column_name.c_str());

    if (!result.aggregation) {
        yardstick_free_measure_agg_result(result);
        throw BinderException("Column '%s' is not a registered measure. Use CREATE VIEW ... AS MEASURE to define measures.", column_name);
    }

    string agg_fn(result.aggregation);
    yardstick_free_measure_agg_result(result);

    // Set return type based on the input type
    bound_function.return_type = arguments[0]->return_type;

    return make_uniq<AggregateBindData>(agg_fn, column_name, bound_function.return_type);
}

// bind_expression callback - transforms AGGREGATE(x) into the actual aggregate function
static unique_ptr<Expression> AggregateBindExpression(FunctionBindExpressionInput &input) {
    if (!input.bind_data) {
        throw InternalException("AGGREGATE bind_expression: missing bind_data");
    }

    auto &bind_data = input.bind_data->Cast<AggregateBindData>();
    auto &children = input.children;

    if (children.empty()) {
        throw InternalException("AGGREGATE bind_expression: no children");
    }

    // Get the aggregate function from the catalog
    auto &context = input.context;

    try {
        auto &catalog = Catalog::GetSystemCatalog(context);
        auto &func_entry = catalog.GetEntry<AggregateFunctionCatalogEntry>(
            context, DEFAULT_SCHEMA, bind_data.agg_function);

        // Get argument types
        vector<LogicalType> arg_types;
        arg_types.push_back(children[0]->return_type);

        // Bind to the correct overload
        FunctionBinder function_binder(context);
        ErrorData error;
        auto best_function = function_binder.BindFunction(
            func_entry.name, func_entry.functions, arg_types, error);

        if (!best_function.IsValid()) {
            throw BinderException("Could not bind aggregate function '%s' for type '%s': %s",
                bind_data.agg_function, arg_types[0].ToString(), error.Message());
        }

        auto bound_function = func_entry.functions.GetFunctionByOffset(best_function.GetIndex());

        // Create the aggregate expression
        vector<unique_ptr<Expression>> agg_children;
        agg_children.push_back(children[0]->Copy());

        auto aggregate = function_binder.BindAggregateFunction(
            bound_function,
            std::move(agg_children),
            nullptr,  // filter
            AggregateType::NON_DISTINCT
        );

        return std::move(aggregate);

    } catch (CatalogException &e) {
        throw BinderException("Aggregate function '%s' not found: %s",
            bind_data.agg_function, e.what());
    }
}

// Dummy execution function - should never be called since bind_expression replaces this
static void AggregateExecuteFunction(DataChunk &args, ExpressionState &state, Vector &result) {
    throw InternalException("AGGREGATE() should have been replaced by bind_expression");
}

//=============================================================================
// PARSER EXTENSION
//=============================================================================

// Check if query starts with SEMANTIC keyword (case insensitive)
static bool StartsWithSemantic(const std::string &query, std::string &stripped_query) {
    // Skip leading whitespace
    size_t start = 0;
    while (start < query.size() && std::isspace(query[start])) {
        start++;
    }

    // Check for "SEMANTIC" prefix (case insensitive)
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

    // Must be followed by whitespace
    if (start + prefix_len < query.size() && !std::isspace(query[start + prefix_len])) {
        return false;
    }

    // Strip the SEMANTIC prefix
    stripped_query = query.substr(start + prefix_len);
    return true;
}

// Check if a string starts with a keyword (case insensitive), skipping whitespace
static bool StartsWithKeyword(const std::string &str, const std::string &keyword, size_t &end_pos) {
    size_t start = 0;
    while (start < str.size() && std::isspace(str[start])) {
        start++;
    }

    if (str.size() - start < keyword.size()) {
        return false;
    }

    for (size_t i = 0; i < keyword.size(); i++) {
        if (std::toupper(str[start + i]) != std::toupper(keyword[i])) {
            return false;
        }
    }

    // Must be followed by whitespace or end of string
    if (start + keyword.size() < str.size() && !std::isspace(str[start + keyword.size()])) {
        return false;
    }

    end_pos = start + keyword.size();
    return true;
}

// Check if query is a CREATE [OR REPLACE] METRIC/DIMENSION/SEGMENT statement
// Returns the statement type or empty string if not matched
// Handles syntaxes like:
//   - "CREATE METRIC name AS expr"
//   - "CREATE OR REPLACE METRIC name AS expr"
//   - "CREATE METRIC (...)"
//   - "CREATE METRIC model.name AS expr"
static std::string IsDefinitionStatement(const std::string &query, std::string &definition, bool &is_replace) {
    size_t pos = 0;
    is_replace = false;

    // Must start with CREATE
    if (!StartsWithKeyword(query, "CREATE", pos)) {
        return "";
    }

    std::string rest = query.substr(pos);

    // Check for optional OR REPLACE
    size_t or_pos = 0;
    if (StartsWithKeyword(rest, "OR", or_pos)) {
        rest = rest.substr(or_pos);
        size_t replace_pos = 0;
        if (StartsWithKeyword(rest, "REPLACE", replace_pos)) {
            rest = rest.substr(replace_pos);
            is_replace = true;
        } else {
            return ""; // "CREATE OR" without "REPLACE" is invalid
        }
    }

    // Helper to check for AS keyword (case-insensitive)
    auto is_as_keyword = [](const std::string &s, size_t p) -> bool {
        if (p + 2 > s.size()) return false;
        return (s[p] == 'A' || s[p] == 'a') &&
               (s[p + 1] == 'S' || s[p + 1] == 's') &&
               (p + 2 >= s.size() || std::isspace(s[p + 2]));
    };

    // Helper lambda to check for definition patterns after keyword
    auto check_definition = [&is_as_keyword](const std::string &after_kw, const std::string &keyword, std::string &def) -> bool {
        size_t start = 0;
        while (start < after_kw.size() && std::isspace(after_kw[start])) start++;

        // Case 1: Direct opening paren - "KEYWORD (..."
        if (start < after_kw.size() && after_kw[start] == '(') {
            def = keyword + " " + after_kw.substr(start);
            return true;
        }

        // Read first identifier
        size_t name_start = start;
        while (start < after_kw.size() && (std::isalnum(after_kw[start]) || after_kw[start] == '_')) start++;
        if (start == name_start) {
            return false;
        }

        // Skip whitespace
        while (start < after_kw.size() && std::isspace(after_kw[start])) start++;

        // Case 2: Check for "AS" keyword - simple SQL syntax "KEYWORD name AS expr"
        if (is_as_keyword(after_kw, start)) {
            def = keyword + " " + after_kw.substr(name_start);
            return true;
        }

        // Case 3: Check for dot (model.name syntax)
        if (start < after_kw.size() && after_kw[start] == '.') {
            start++; // skip dot
            // Read second identifier (field name)
            size_t field_start = start;
            while (start < after_kw.size() && (std::isalnum(after_kw[start]) || after_kw[start] == '_')) start++;
            if (start == field_start) {
                return false;
            }
            // Skip whitespace
            while (start < after_kw.size() && std::isspace(after_kw[start])) start++;

            // Check for AS (model.name AS expr syntax)
            if (is_as_keyword(after_kw, start)) {
                def = keyword + " " + after_kw.substr(name_start);
                return true;
            }

            // Check for paren (model.name (props) syntax)
            if (start < after_kw.size() && after_kw[start] == '(') {
                def = keyword + " " + after_kw.substr(name_start);
                return true;
            }
        }

        return false;
    };

    // Check for METRIC
    size_t metric_pos = 0;
    if (StartsWithKeyword(rest, "METRIC", metric_pos)) {
        std::string after_metric = rest.substr(metric_pos);
        if (check_definition(after_metric, "METRIC", definition)) {
            return "METRIC";
        }
    }

    // Check for DIMENSION
    size_t dim_pos = 0;
    if (StartsWithKeyword(rest, "DIMENSION", dim_pos)) {
        std::string after_dim = rest.substr(dim_pos);
        if (check_definition(after_dim, "DIMENSION", definition)) {
            return "DIMENSION";
        }
    }

    // Check for SEGMENT
    size_t seg_pos = 0;
    if (StartsWithKeyword(rest, "SEGMENT", seg_pos)) {
        std::string after_seg = rest.substr(seg_pos);
        if (check_definition(after_seg, "SEGMENT", definition)) {
            return "SEGMENT";
        }
    }

    return "";
}

// Check if stripped query is a CREATE [OR REPLACE] MODEL statement
// Returns: 0 = not a create model, 1 = create model, 2 = create or replace model
// Sets definition to be in nom-parser format: "MODEL (name ..., ...)"
static int IsCreateModelStatement(const std::string &query, std::string &definition) {
    size_t pos = 0;

    // Check for CREATE
    if (!StartsWithKeyword(query, "CREATE", pos)) {
        return 0;
    }

    std::string rest = query.substr(pos);
    bool is_replace = false;

    // Check for optional OR REPLACE
    size_t or_pos = 0;
    if (StartsWithKeyword(rest, "OR", or_pos)) {
        rest = rest.substr(or_pos);
        size_t replace_pos = 0;
        if (StartsWithKeyword(rest, "REPLACE", replace_pos)) {
            rest = rest.substr(replace_pos);
            is_replace = true;
        } else {
            return 0; // "CREATE OR" without "REPLACE" is invalid
        }
    }

    // Check for MODEL
    size_t model_pos = 0;
    if (!StartsWithKeyword(rest, "MODEL", model_pos)) {
        return 0;
    }

    // Skip whitespace after MODEL
    rest = rest.substr(model_pos);
    size_t start = 0;
    while (start < rest.size() && std::isspace(rest[start])) {
        start++;
    }
    rest = rest.substr(start);

    // Find the opening parenthesis - everything from there is the definition body
    size_t paren_pos = rest.find('(');
    if (paren_pos == std::string::npos) {
        return 0; // No parenthesis found
    }

    // Build definition in nom format: "MODEL (name ..., ...)"
    // The content inside parens should already have "name xxx" as first property
    definition = "MODEL " + rest.substr(paren_pos);
    return is_replace ? 2 : 1;
}

// Global to store database path for parser extension (set during extension load)
static std::string g_db_path;

ParserExtensionParseResult yardstick_parse(ParserExtensionInfo *,
                                            const std::string &query) {

    // ===========================================================================
    // Julian Hyde "Measures in SQL" (arXiv:2406.00251)
    // Check for AS MEASURE and AGGREGATE() in SQL (with or without SEMANTIC prefix)
    // ===========================================================================

    // Determine the SQL to check for measures (strip SEMANTIC prefix if present)
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

            // Always wrap AGGREGATE queries in table function to avoid binding issues.
            // Direct binding through the extension causes DuckDB internal errors
            // ("Unsupported type for NumericValueUnionToValue").
            // Escape single quotes in the SQL for embedding in string literal
            string escaped_sql;
            for (char c : expanded_sql) {
                if (c == '\'') {
                    escaped_sql += "''";
                } else {
                    escaped_sql += c;
                }
            }

            // Wrap in table function call - this bypasses the problematic extension binding
            string wrapper_sql = "SELECT * FROM yardstick('" + escaped_sql + "')";

            Parser parser;
            parser.ParseQuery(wrapper_sql);
            auto statements = std::move(parser.statements);

            if (statements.empty()) {
                return ParserExtensionParseResult("Table function wrapper produced no statements");
            }

            // Return without going through extension - DuckDB binds table function normally
            return ParserExtensionParseResult(
                make_uniq_base<ParserExtensionParseData, YardstickParseData>(
                    std::move(statements[0])));
        }

        yardstick_free_aggregate_result(result);
    }

    // Check for CREATE VIEW with AS MEASURE (regular SQL)
    if (yardstick_has_as_measure(sql_to_check.c_str())) {
        YardstickCreateViewResult result = yardstick_process_create_view(query.c_str());

        if (result.error) {
            string error_msg(result.error);
            yardstick_free_create_view_result(result);
            return ParserExtensionParseResult(error_msg);
        }

        if (result.is_measure_view) {
            // Parse the clean SQL (without AS MEASURE) and execute it
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

    // ===========================================================================
    // Original SEMANTIC prefix handling
    // ===========================================================================

    // Check for SEMANTIC prefix
    std::string stripped_query;
    if (!StartsWithSemantic(query, stripped_query)) {
        // Not a semantic query, let DuckDB handle it
        return ParserExtensionParseResult();
    }

    // Check if this is a CREATE [OR REPLACE] MODEL statement
    std::string definition;
    int create_type = IsCreateModelStatement(stripped_query, definition);

    if (create_type > 0) {
        // This is a CREATE MODEL statement - handle specially
        bool replace = (create_type == 2);
        const char *db_path_ptr = g_db_path.empty() ? nullptr : g_db_path.c_str();

        char *error = yardstick_define(definition.c_str(), db_path_ptr, replace);
        if (error) {
            string error_msg(error);
            yardstick_free(error);
            return ParserExtensionParseResult(error_msg);
        }

        // Return a simple SELECT statement as acknowledgment
        Parser parser;
        parser.ParseQuery("SELECT 'Model created successfully' AS result");
        auto statements = std::move(parser.statements);

        return ParserExtensionParseResult(
            make_uniq_base<ParserExtensionParseData, YardstickParseData>(
                std::move(statements[0])));
    }

    // Check if this is a MODEL <model> statement (to switch active model)
    // Using "SEMANTIC MODEL orders" instead of "SEMANTIC USE orders" because DuckDB
    // handles USE statements specially before parser extensions are called
    size_t model_pos = 0;
    if (StartsWithKeyword(stripped_query, "MODEL", model_pos)) {
        std::string rest = stripped_query.substr(model_pos);
        // Trim whitespace and get model name (until semicolon, paren, or end)
        size_t start = 0;
        while (start < rest.size() && std::isspace(rest[start])) start++;
        size_t end = start;
        while (end < rest.size() && !std::isspace(rest[end]) && rest[end] != ';' && rest[end] != '(') end++;
        std::string model_name = rest.substr(start, end - start);

        // Skip if there's a paren after - that's CREATE MODEL syntax
        size_t paren_check = end;
        while (paren_check < rest.size() && std::isspace(rest[paren_check])) paren_check++;
        if (paren_check < rest.size() && rest[paren_check] == '(') {
            // This looks like a CREATE MODEL with inline parens, skip
            goto not_model_switch;
        }

        if (!model_name.empty()) {
            char *error = yardstick_use(model_name.c_str());
            if (error) {
                string error_msg(error);
                yardstick_free(error);
                return ParserExtensionParseResult(error_msg);
            }

            // Return acknowledgment
            Parser parser;
            parser.ParseQuery("SELECT 'Using model: " + model_name + "' AS result");
            auto statements = std::move(parser.statements);

            return ParserExtensionParseResult(
                make_uniq_base<ParserExtensionParseData, YardstickParseData>(
                    std::move(statements[0])));
        }
    }

not_model_switch:
    // Check if this is a CREATE [OR REPLACE] METRIC/DIMENSION/SEGMENT statement
    bool is_replace = false;
    std::string def_type = IsDefinitionStatement(stripped_query, definition, is_replace);
    if (!def_type.empty()) {
        const char *db_path_ptr = g_db_path.empty() ? nullptr : g_db_path.c_str();

        char *error = yardstick_add_definition(definition.c_str(), db_path_ptr, is_replace);
        if (error) {
            string error_msg(error);
            yardstick_free(error);
            return ParserExtensionParseResult(error_msg);
        }

        // Return acknowledgment
        Parser parser;
        std::string action = is_replace ? "replaced" : "created";
        parser.ParseQuery("SELECT '" + def_type + " " + action + " successfully' AS result");
        auto statements = std::move(parser.statements);

        return ParserExtensionParseResult(
            make_uniq_base<ParserExtensionParseData, YardstickParseData>(
                std::move(statements[0])));
    }

    // Regular SEMANTIC SELECT query - try to rewrite using yardstick
    YardstickRewriteResult result = yardstick_rewrite(stripped_query.c_str());

    // If there was an error, return it
    if (result.error) {
        string error_msg(result.error);
        yardstick_free_result(result);
        return ParserExtensionParseResult(error_msg);
    }

    // Parse the rewritten SQL using DuckDB's parser
    string rewritten_sql(result.sql);
    yardstick_free_result(result);

    Parser parser;
    parser.ParseQuery(rewritten_sql);
    auto statements = std::move(parser.statements);

    if (statements.empty()) {
        return ParserExtensionParseResult("Rewritten query produced no statements");
    }

    // Return the parsed statement
    return ParserExtensionParseResult(
        make_uniq_base<ParserExtensionParseData, YardstickParseData>(
            std::move(statements[0])));
}

ParserExtensionPlanResult yardstick_plan(ParserExtensionInfo *,
                                          ClientContext &context,
                                          unique_ptr<ParserExtensionParseData> parse_data) {
    // Store parse data in client context state
    auto state = make_shared_ptr<YardstickState>(std::move(parse_data));
    context.registered_state->Remove("yardstick");
    context.registered_state->Insert("yardstick", state);

    // Throw to trigger the operator extension's Bind function
    throw BinderException("Use yardstick_bind instead");
}

BoundStatement yardstick_bind(ClientContext &context, Binder &binder,
                               OperatorExtensionInfo *info, SQLStatement &statement) {
    switch (statement.type) {
    case StatementType::EXTENSION_STATEMENT: {
        auto &ext_statement = dynamic_cast<ExtensionStatement &>(statement);

        // Check this is our extension's statement
        if (ext_statement.extension.parse_function == yardstick_parse) {
            // Retrieve stashed parse data
            auto lookup = context.registered_state->Get<YardstickState>("yardstick");
            if (lookup) {
                auto state = (YardstickState *)lookup.get();
                auto parse_data = dynamic_cast<YardstickParseData *>(state->parse_data.get());

                // For SELECT statements with correlated subqueries (from AT SET expansion),
                // use a fresh binder to avoid binding conflicts. For DDL statements
                // (CREATE VIEW), keep the parent for transaction context.
                shared_ptr<Binder> yardstick_binder;
                if (parse_data->statement->type == StatementType::SELECT_STATEMENT) {
                    yardstick_binder = Binder::CreateBinder(context);
                } else {
                    yardstick_binder = Binder::CreateBinder(context, &binder);
                }

                // Bind the SQL statement we generated
                return yardstick_binder->Bind(*(parse_data->statement));
            }
            throw BinderException("Registered state not found");
        }
    }
    default:
        return {};  // Not ours
    }
}

//=============================================================================
// CATALOG SCHEMA INITIALIZATION
//=============================================================================

static const char *YARDSTICK_SCHEMA_DDL = R"(
CREATE SCHEMA IF NOT EXISTS yardstick;

CREATE TABLE IF NOT EXISTS yardstick.models (
    model_id INTEGER PRIMARY KEY,
    name VARCHAR UNIQUE NOT NULL,
    table_name VARCHAR NOT NULL,
    sql_definition VARCHAR,
    primary_key VARCHAR NOT NULL,
    description VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS yardstick.dimensions (
    dimension_id INTEGER PRIMARY KEY,
    model_id INTEGER REFERENCES yardstick.models(model_id),
    name VARCHAR NOT NULL,
    type VARCHAR DEFAULT 'categorical',
    sql_expression VARCHAR,
    granularity VARCHAR,
    description VARCHAR,
    UNIQUE (model_id, name)
);

CREATE TABLE IF NOT EXISTS yardstick.measures (
    measure_id INTEGER PRIMARY KEY,
    model_id INTEGER REFERENCES yardstick.models(model_id),
    name VARCHAR NOT NULL,
    type VARCHAR DEFAULT 'simple',
    aggregation VARCHAR,
    sql_expression VARCHAR,
    numerator VARCHAR,
    denominator VARCHAR,
    description VARCHAR,
    UNIQUE (model_id, name)
);

CREATE TABLE IF NOT EXISTS yardstick.relationships (
    relationship_id INTEGER PRIMARY KEY,
    model_id INTEGER REFERENCES yardstick.models(model_id),
    target_model VARCHAR NOT NULL,
    type VARCHAR DEFAULT 'many_to_one',
    foreign_key VARCHAR,
    primary_key VARCHAR DEFAULT 'id',
    sql_condition VARCHAR,
    UNIQUE (model_id, target_model)
);

CREATE TABLE IF NOT EXISTS yardstick.segments (
    segment_id INTEGER PRIMARY KEY,
    model_id INTEGER REFERENCES yardstick.models(model_id),
    name VARCHAR NOT NULL,
    sql_condition VARCHAR NOT NULL,
    description VARCHAR,
    UNIQUE (model_id, name)
);

CREATE TABLE IF NOT EXISTS yardstick.measure_views (
    view_id INTEGER PRIMARY KEY,
    view_name VARCHAR UNIQUE NOT NULL,
    base_query VARCHAR NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS yardstick.view_measures (
    view_measure_id INTEGER PRIMARY KEY,
    view_id INTEGER REFERENCES yardstick.measure_views(view_id),
    column_name VARCHAR NOT NULL,
    measure_expression VARCHAR NOT NULL,
    UNIQUE (view_id, column_name)
);

CREATE SEQUENCE IF NOT EXISTS yardstick.model_id_seq START 1;
CREATE SEQUENCE IF NOT EXISTS yardstick.dimension_id_seq START 1;
CREATE SEQUENCE IF NOT EXISTS yardstick.measure_id_seq START 1;
CREATE SEQUENCE IF NOT EXISTS yardstick.relationship_id_seq START 1;
CREATE SEQUENCE IF NOT EXISTS yardstick.segment_id_seq START 1;
CREATE SEQUENCE IF NOT EXISTS yardstick.view_id_seq START 1;
CREATE SEQUENCE IF NOT EXISTS yardstick.view_measure_id_seq START 1;
)";

static void InitializeCatalogSchema(DatabaseInstance &db) {
    Connection con(db);

    // Split DDL by semicolons and execute each statement
    std::string ddl(YARDSTICK_SCHEMA_DDL);
    std::istringstream stream(ddl);
    std::string statement;

    while (std::getline(stream, statement, ';')) {
        // Trim whitespace
        size_t start = statement.find_first_not_of(" \t\n\r");
        if (start == std::string::npos) continue;
        size_t end = statement.find_last_not_of(" \t\n\r");
        statement = statement.substr(start, end - start + 1);

        if (!statement.empty()) {
            auto result = con.Query(statement);
            if (result->HasError()) {
                // Log but don't fail - schema might already exist
                // fprintf(stderr, "Warning: Schema init error: %s\n", result->GetError().c_str());
            }
        }
    }
}

//=============================================================================
// EXTENSION LOADING
//=============================================================================

static void LoadInternal(ExtensionLoader &loader) {
    auto &db = loader.GetDatabaseInstance();
    auto &config = DBConfig::GetConfig(db);

    // Capture database path for CREATE MODEL statements
    auto &db_config = db.config;
    if (!db_config.options.database_path.empty()) {
        g_db_path = db_config.options.database_path;
    } else {
        g_db_path.clear();
    }

    // Initialize catalog schema (creates yardstick.* tables if they don't exist)
    InitializeCatalogSchema(db);

    // Auto-load definitions from file if it exists
    const char *db_path_ptr = g_db_path.empty() ? nullptr : g_db_path.c_str();
    char *error = yardstick_autoload(db_path_ptr);
    if (error) {
        // Log warning but don't fail extension load
        // fprintf(stderr, "Warning: failed to autoload yardstick definitions: %s\n", error);
        yardstick_free(error);
    }

    // Register parser extension
    YardstickParserExtension parser;
    config.parser_extensions.push_back(parser);

    // Register operator extension
    config.operator_extensions.push_back(make_uniq<YardstickOperatorExtension>());

    // Register table functions
    TableFunction query_func("yardstick", {LogicalType::VARCHAR},
                             YardstickQueryFunction, YardstickQueryBind);
    loader.RegisterFunction(query_func);

    TableFunction load_func("yardstick_load", {LogicalType::VARCHAR},
                            YardstickLoadFunction, YardstickLoadBind);
    loader.RegisterFunction(load_func);

    TableFunction load_file_func("yardstick_load_file", {LogicalType::VARCHAR},
                                  YardstickLoadFileFunction, YardstickLoadFileBind);
    loader.RegisterFunction(load_file_func);

    TableFunction models_func("yardstick_models", {},
                              YardstickModelsFunction, YardstickModelsBind);
    loader.RegisterFunction(models_func);

    // Register scalar function for manual rewriting
    auto rewrite_func = ScalarFunction("yardstick_rewrite_sql",
                                        {LogicalType::VARCHAR},
                                        LogicalType::VARCHAR,
                                        YardstickRewriteSqlFunction);
    loader.RegisterFunction(rewrite_func);

    // NOTE: We intentionally do NOT register an AGGREGATE function here.
    // Instead, we rely on parser-level rewriting when AS MEASURE views are queried.
    // For now, users must use SEMANTIC prefix: SEMANTIC SELECT AGGREGATE(revenue) FROM view
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
