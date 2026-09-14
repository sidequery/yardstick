#define DUCKDB_EXTENSION_MAIN

#include "yardstick_extension.hpp"
#include "yardstick_parser_extension.hpp"
#include "frontend_peg.hpp"
#include "aggregate_state.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/parser/parser_extension.hpp"
#include "duckdb/parser/statement/extension_statement.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/connection.hpp"
#if YARDSTICK_GRAMMAR_EXTENSION
#include "duckdb/main/client_config.hpp"
#include "duckdb/parser/statement/create_statement.hpp"
#include "duckdb/parser/parsed_data/create_view_info.hpp"
#include "duckdb/parser/statement/select_statement.hpp"
#include "duckdb/parser/tableref/subqueryref.hpp"
#include "duckdb/planner/operator/logical_create.hpp"
#include <atomic>
#include <map>
#endif
#include "duckdb/logging/logger.hpp"

#include <type_traits>
#include <utility>

// Include FFI types header
#include "yardstick_ffi.h"

// Forward declare C++ parser FFI functions (defined in yardstick_parser_ffi.cpp)
extern "C" {
    YardstickAggregateCallList* yardstick_find_aggregates(const char* sql);
    void yardstick_free_aggregate_list(YardstickAggregateCallList* list);
    YardstickSelectInfo* yardstick_parse_select(const char* sql);
    void yardstick_free_select_info(YardstickSelectInfo* info);
    YardstickExpressionInfo* yardstick_parse_expression(const char* expr);
    void yardstick_free_expression_info(YardstickExpressionInfo* info);
    YardstickCreateViewInfo* yardstick_parse_create_view(const char* sql);
    void yardstick_free_create_view_info(YardstickCreateViewInfo* info);
    char* yardstick_replace_range(const char* sql, uint32_t start, uint32_t end, const char* replacement);
    char* yardstick_apply_replacements(const char* sql, const YardstickReplacement* replacements, size_t count);
    char* yardstick_qualify_expression(const char* expr, const char* qualifier, const char* dimension);
    void yardstick_free_string(char* ptr);
    char* yardstick_expand_aggregate_call(
        const char* measure_name,
        const char* agg_func,
        const YardstickAtModifier* modifiers,
        size_t modifier_count,
        const char* table_name,
        const char* outer_alias,
        const char* outer_where,
        const char* const* group_by_cols,
        size_t group_by_count
    );
}

// Rust FFI - Julian Hyde "Measures in SQL" (arXiv:2406.00251)
// Note: Struct types (YardstickCreateViewResult, YardstickAggregateResult, etc.)
// are already defined in yardstick_ffi.h
extern "C" {
    char *yardstick_rewrite_percentile_within_group(const char *sql);
    bool yardstick_has_as_measure(const char *sql);
    bool yardstick_has_aggregate(const char *sql);
    bool yardstick_drop_measure_view_from_sql(const char *sql);
    char *yardstick_extract_view_name(const char *sql);
    char *yardstick_extract_drop_view_name(const char *sql);
    void *yardstick_snapshot_measure_view(const char *view_name);
    void yardstick_push_measure_view_overlay();
    void yardstick_pop_measure_view_overlay();
    void yardstick_set_measure_view_overlay(const char *name, const void *snapshot);
    void yardstick_bypass_measure_view_overlay(const char *name);
    void *yardstick_empty_measure_view_snapshot();
    void yardstick_restore_measure_view_snapshot(const char *view_name, void *snapshot);
    void yardstick_free_measure_view_snapshot(void *snapshot);
    YardstickCreateViewResult yardstick_process_create_view(const char *sql);
    YardstickAggregateResult yardstick_expand_aggregate(const char *sql);
    void yardstick_free(char *ptr);
    void yardstick_free_create_view_result(YardstickCreateViewResult result);
    void yardstick_free_aggregate_result(YardstickAggregateResult result);

    // Initialize parser FFI function pointers in Rust
    void yardstick_init_parser_ffi(
        YardstickAggregateCallList* (*find_aggregates)(const char*),
        void (*free_aggregate_list)(YardstickAggregateCallList*),
        YardstickSelectInfo* (*parse_select)(const char*),
        void (*free_select_info)(YardstickSelectInfo*),
        YardstickExpressionInfo* (*parse_expression)(const char*),
        void (*free_expression_info)(YardstickExpressionInfo*),
        YardstickCreateViewInfo* (*parse_create_view)(const char*),
        void (*free_create_view_info)(YardstickCreateViewInfo*),
        char* (*replace_range)(const char*, uint32_t, uint32_t, const char*),
        char* (*apply_replacements)(const char*, const YardstickReplacement*, size_t),
        char* (*qualify_expression)(const char*, const char*, const char*),
        char* (*inline_order_by_subquery_aliases)(const char*),
        void (*free_string)(char*),
        char* (*expand_aggregate_call)(const char*, const char*, const YardstickAtModifier*, size_t, const char*, const char*, const char*, const char* const*, size_t),
        YardstickCurrentReferenceList* (*find_current_references)(const char*),
        void (*free_current_references)(YardstickCurrentReferenceList*),
        int32_t (*current_where_is_single_valued)(const char*, const char*, const char*),
        int32_t (*expressions_equal)(const char*, const char*),
        YardstickQueryScopeList* (*find_query_scopes)(const char*),
        void (*free_query_scopes)(YardstickQueryScopeList*),
        char* (*rewrite_visible_filter)(const char*, const char*, const char* const*, const char* const*, size_t, char**),
        char* (*decorate_measure)(const char*, const char*, const char* const*, const char* const*, size_t,
                                  const char* const*, size_t, char**),
        char* (*window_marker)(const char*, const char*, char**),
        char* (*rewrite_measure_windows)(const char*, const YardstickWindowSource*, size_t,
                                         const YardstickWindowCall*, size_t, const char* const*, size_t, char**)
    );
}

namespace duckdb {

// Rewrites must preserve the caller's grammar while avoiding parser-override
// recursion. Outside an override, legacy callers retain the default parser.
static ParserOptions YardstickParserOptions() {
#if YARDSTICK_GRAMMAR_EXTENSION
    if (auto *options = CurrentNativeYardstickParserOptions()) {
        return *options;
    }
#endif
    return ParserOptions();
}

static std::string RewritePercentileWithinGroup(const std::string &sql) {
    // Canonicalize before registration so stored expressions and executable SQL
    // share the same percentile lowering on every supported DuckDB version.
    char *rewritten = yardstick_rewrite_percentile_within_group(sql.c_str());
    if (!rewritten) {
        throw InvalidInputException("Failed to normalize percentile expressions");
    }
    std::string result(rewritten);
    yardstick_free(rewritten);
    return result;
}

//=============================================================================
// TABLE FUNCTION: yardstick(sql) - Execute SQL with AGGREGATE() expansion
//=============================================================================

static std::string AggregateWarnings(YardstickAggregateResult &result);
static void HandleAggregateWarnings(ClientContext &context, const string &warnings);

// DuckDB 2.0 uses Identifier for result and table-function column names.
using YardstickColumnName = std::decay<decltype(std::declval<QueryResult>().ColumnName(0))>::type;

struct YardstickQueryData : public TableFunctionData {
    string original_sql;
    string rewritten_sql;
    unique_ptr<QueryResult> result;
    bool done = false;
};

#if YARDSTICK_GRAMMAR_EXTENSION
struct DeferredMeasureFunctionInfo : public TableFunctionInfo {
    explicit DeferredMeasureFunctionInfo(shared_ptr<ParserExtensionInfo> parser_info_p)
        : parser_info(std::move(parser_info_p)) {
    }
    shared_ptr<ParserExtensionInfo> parser_info;
};
#endif

static unique_ptr<FunctionData> YardstickQueryBind(ClientContext &context,
                                                     TableFunctionBindInput &input,
                                                     vector<LogicalType> &return_types,
                                                     vector<YardstickColumnName> &names) {
#if YARDSTICK_GRAMMAR_EXTENSION
    NativeYardstickBindScope bind_scope(context);
    auto &info = input.info->Cast<DeferredMeasureFunctionInfo>();
    NativeYardstickParseScope native_scope(info.parser_info.get(), context.GetParserOptions());
#endif
    auto data = make_uniq<YardstickQueryData>();
    data->original_sql = input.inputs[0].GetValue<string>();
    if (input.inputs.size() > 1) {
        HandleAggregateWarnings(context, input.inputs[1].GetValue<string>());
    }

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
            HandleAggregateWarnings(context, AggregateWarnings(result));
        } else {
            data->rewritten_sql = data->original_sql;
        }
        yardstick_free_aggregate_result(result);
    } else {
        data->rewritten_sql = data->original_sql;
    }

    // Execute the rewritten query to get schema
    Connection con(*context.db);
#if YARDSTICK_GRAMMAR_EXTENSION
    // The wrapper executes on another connection; preserve the caller's grammar
    // there too, without changing the caller's activation settings.
    auto &source_config = ClientConfig::GetConfig(context);
    auto &execution_config = ClientConfig::GetConfig(*con.context);
    execution_config.active_grammar_extensions = source_config.active_grammar_extensions;
    execution_config.cached_grammar = source_config.cached_grammar;
#endif
    auto query_result = con.Query(data->rewritten_sql);

    if (query_result->HasError()) {
        throw InvalidInputException("Query error: %s", query_result->GetError());
    }

    // Extract return types and names from result
    for (idx_t i = 0; i < query_result->ColumnCount(); i++) {
#if YARDSTICK_RESULT_ACCESSORS
        return_types.push_back(query_result->GetTypes()[i]);
#else
        return_types.push_back(query_result->types[i]);
#endif
        names.push_back(query_result->ColumnName(i));
    }
    // Table bindings require unique names even when the executed SELECT returns
    // duplicate aliases. Use DuckDB's suffix policy without flattening Identifiers.
    QueryResult::DeduplicateColumns(names);

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

static void YardstickWarningFunction(DataChunk &input, ExpressionState &state, Vector &result) {
    if (input.ColumnCount() > 0 && input.size() > 0) {
        auto warnings = input.data[0].GetValue(0);
        if (!warnings.IsNull()) {
            HandleAggregateWarnings(state.GetContext(), warnings.GetValue<string>());
        }
    }
    result.SetVectorType(VectorType::CONSTANT_VECTOR);
    ConstantVector::SetNull(result, false);
    ConstantVector::GetData<bool>(result)[0] = true;
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

static std::vector<std::string> SplitSqlStatements(const std::string &sql) {
    std::vector<std::string> statements;
    size_t statement_start = 0;
    std::string dollar_quote_end;
    bool in_single_quote = false;
    bool in_double_quote = false;
    bool in_line_comment = false;
    bool in_block_comment = false;

    for (size_t i = 0; i < sql.size(); i++) {
        char c = sql[i];
        char next = i + 1 < sql.size() ? sql[i + 1] : '\0';

        if (in_line_comment) {
            if (c == '\n') {
                in_line_comment = false;
            }
            continue;
        }

        if (in_block_comment) {
            if (c == '*' && next == '/') {
                in_block_comment = false;
                i++;
            }
            continue;
        }

        if (!dollar_quote_end.empty()) {
            if (sql.compare(i, dollar_quote_end.size(), dollar_quote_end) == 0) {
                i += dollar_quote_end.size() - 1;
                dollar_quote_end.clear();
            }
            continue;
        }

        if (in_single_quote) {
            if (c == '\'' && next == '\'') {
                i++;
            } else if (c == '\'') {
                in_single_quote = false;
            }
            continue;
        }

        if (in_double_quote) {
            if (c == '"' && next == '"') {
                i++;
            } else if (c == '"') {
                in_double_quote = false;
            }
            continue;
        }

        if (c == '-' && next == '-') {
            in_line_comment = true;
            i++;
            continue;
        }
        if (c == '/' && next == '*') {
            in_block_comment = true;
            i++;
            continue;
        }
        if (c == '\'') {
            in_single_quote = true;
            continue;
        }
        if (c == '"') {
            in_double_quote = true;
            continue;
        }
        if (c == '$') {
            size_t tag_end = i + 1;
            while (tag_end < sql.size() && sql[tag_end] != '$') {
                char tag_char = sql[tag_end];
                if (!std::isalnum(static_cast<unsigned char>(tag_char)) && tag_char != '_') {
                    break;
                }
                tag_end++;
            }
            if (tag_end < sql.size() && sql[tag_end] == '$') {
                dollar_quote_end = sql.substr(i, tag_end - i + 1);
                i = tag_end;
                continue;
            }
        }

        if (c == ';') {
            statements.push_back(sql.substr(statement_start, i - statement_start));
            statement_start = i + 1;
        }
    }

    statements.push_back(sql.substr(statement_start));
    return statements;
}

static bool IsIdentifierChar(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
}

static size_t SkipWhitespaceAndComments(const std::string &sql, size_t pos) {
    while (pos < sql.size()) {
        if (std::isspace(static_cast<unsigned char>(sql[pos]))) {
            pos++;
            continue;
        }
        if (pos + 1 < sql.size() && sql[pos] == '-' && sql[pos + 1] == '-') {
            pos += 2;
            while (pos < sql.size() && sql[pos] != '\n') {
                pos++;
            }
            continue;
        }
        if (pos + 1 < sql.size() && sql[pos] == '/' && sql[pos + 1] == '*') {
            pos += 2;
            while (pos + 1 < sql.size() && !(sql[pos] == '*' && sql[pos + 1] == '/')) {
                pos++;
            }
            pos = std::min(pos + 2, sql.size());
            continue;
        }
        break;
    }
    return pos;
}

static bool ConsumeKeyword(const std::string &sql, size_t &pos, const char *keyword) {
    size_t keyword_len = strlen(keyword);
    if (pos + keyword_len > sql.size()) {
        return false;
    }
    for (size_t i = 0; i < keyword_len; i++) {
        if (std::toupper(static_cast<unsigned char>(sql[pos + i])) != keyword[i]) {
            return false;
        }
    }
    if (pos + keyword_len < sql.size() && IsIdentifierChar(sql[pos + keyword_len])) {
        return false;
    }
    pos += keyword_len;
    return true;
}

static bool StartsWithCreateViewStatement(const std::string &sql) {
    size_t pos = SkipWhitespaceAndComments(sql, 0);
    if (!ConsumeKeyword(sql, pos, "CREATE")) {
        return false;
    }

    pos = SkipWhitespaceAndComments(sql, pos);
    if (ConsumeKeyword(sql, pos, "OR")) {
        pos = SkipWhitespaceAndComments(sql, pos);
        if (!ConsumeKeyword(sql, pos, "REPLACE")) {
            return false;
        }
        pos = SkipWhitespaceAndComments(sql, pos);
    }

    if (ConsumeKeyword(sql, pos, "TEMPORARY") ||
        ConsumeKeyword(sql, pos, "TEMP")) {
        pos = SkipWhitespaceAndComments(sql, pos);
    }

    return ConsumeKeyword(sql, pos, "VIEW");
}

static bool IsTemporaryCreateViewStatement(const std::string &sql) {
    size_t pos = SkipWhitespaceAndComments(sql, 0);
    if (!ConsumeKeyword(sql, pos, "CREATE")) {
        return false;
    }

    pos = SkipWhitespaceAndComments(sql, pos);
    if (ConsumeKeyword(sql, pos, "OR")) {
        pos = SkipWhitespaceAndComments(sql, pos);
        if (!ConsumeKeyword(sql, pos, "REPLACE")) {
            return false;
        }
        pos = SkipWhitespaceAndComments(sql, pos);
    }

    return ConsumeKeyword(sql, pos, "TEMPORARY") ||
           ConsumeKeyword(sql, pos, "TEMP");
}

static bool StartsWithDropViewStatement(const std::string &sql) {
    size_t pos = SkipWhitespaceAndComments(sql, 0);
    if (!ConsumeKeyword(sql, pos, "DROP")) {
        return false;
    }
    pos = SkipWhitespaceAndComments(sql, pos);
    return ConsumeKeyword(sql, pos, "VIEW");
}

static bool StartsWithSelectStatement(const std::string &sql) {
    size_t pos = SkipWhitespaceAndComments(sql, 0);
    return ConsumeKeyword(sql, pos, "SELECT");
}

struct MeasureRewriteResult {
    bool had_measure_view = false;
    bool had_aggregate = false;
    std::string rewritten_sql;
    std::string error;
};

struct MeasureViewSnapshot {
    std::string view_name;
    void *snapshot = nullptr;
};

static void RestoreMeasureViewSnapshots(std::vector<MeasureViewSnapshot> &snapshots) {
    for (auto it = snapshots.rbegin(); it != snapshots.rend(); ++it) {
        yardstick_restore_measure_view_snapshot(it->view_name.c_str(), it->snapshot);
    }
    snapshots.clear();
}

static void FreeMeasureViewSnapshots(std::vector<MeasureViewSnapshot> &snapshots) {
    for (auto &snapshot : snapshots) {
        yardstick_free_measure_view_snapshot(snapshot.snapshot);
    }
    snapshots.clear();
}

static MeasureViewSnapshot SnapshotMeasureView(const std::string &view_name) {
    return {view_name, yardstick_snapshot_measure_view(view_name.c_str())};
}

static bool EqualsCaseInsensitive(const std::string &left, const std::string &right) {
    if (left.size() != right.size()) {
        return false;
    }
    for (idx_t i = 0; i < left.size(); i++) {
        if (std::tolower(static_cast<unsigned char>(left[i])) !=
            std::tolower(static_cast<unsigned char>(right[i]))) {
            return false;
        }
    }
    return true;
}

static bool RemoveCaseInsensitive(std::vector<std::string> &values, const std::string &value) {
    bool removed = false;
    for (auto it = values.begin(); it != values.end();) {
        if (EqualsCaseInsensitive(*it, value)) {
            it = values.erase(it);
            removed = true;
        } else {
            ++it;
        }
    }
    return removed;
}

static bool ContainsCaseInsensitive(const std::vector<std::string> &values, const std::string &value) {
    for (auto &candidate : values) {
        if (EqualsCaseInsensitive(candidate, value)) {
            return true;
        }
    }
    return false;
}

static bool RestoreAndRemoveTemporarySnapshot(std::vector<MeasureViewSnapshot> &snapshots,
                                              const std::string &view_name) {
    bool restored = false;
    for (auto it = snapshots.begin(); it != snapshots.end();) {
        if (EqualsCaseInsensitive(it->view_name, view_name)) {
            if (!restored) {
                yardstick_restore_measure_view_snapshot(it->view_name.c_str(), it->snapshot);
                restored = true;
            } else {
                yardstick_free_measure_view_snapshot(it->snapshot);
            }
            it = snapshots.erase(it);
        } else {
            ++it;
        }
    }
    return restored;
}

static bool HasMeasureSnapshot(const std::vector<MeasureViewSnapshot> &snapshots,
                               const std::string &view_name) {
    for (auto &snapshot : snapshots) {
        if (EqualsCaseInsensitive(snapshot.view_name, view_name)) {
            return true;
        }
    }
    return false;
}

static bool TransferTemporarySnapshotForPermanentDrop(std::vector<MeasureViewSnapshot> &temporary_snapshots,
                                                      const std::string &view_name,
                                                      std::vector<MeasureViewSnapshot> &rollback_snapshots) {
    for (auto &snapshot : temporary_snapshots) {
        if (EqualsCaseInsensitive(snapshot.view_name, view_name)) {
            rollback_snapshots.push_back({snapshot.view_name, snapshot.snapshot});
            snapshot.snapshot = yardstick_empty_measure_view_snapshot();
            return true;
        }
    }
    return false;
}

static bool RestoreShadowedPermanentMetadata(std::vector<MeasureViewSnapshot> &temporary_snapshots,
                                             const std::string &view_name) {
    for (auto &snapshot : temporary_snapshots) {
        if (EqualsCaseInsensitive(snapshot.view_name, view_name)) {
            void *permanent_snapshot = snapshot.snapshot;
            snapshot.snapshot = yardstick_snapshot_measure_view(snapshot.view_name.c_str());
            yardstick_restore_measure_view_snapshot(snapshot.view_name.c_str(), permanent_snapshot);
            return true;
        }
    }
    return false;
}

static void RestoreTemporaryMetadata(std::vector<MeasureViewSnapshot> &temporary_snapshots,
                                     const std::vector<std::string> &view_names) {
    for (auto &view_name : view_names) {
        for (auto &snapshot : temporary_snapshots) {
            if (EqualsCaseInsensitive(snapshot.view_name, view_name)) {
                void *permanent_snapshot = yardstick_snapshot_measure_view(snapshot.view_name.c_str());
                yardstick_restore_measure_view_snapshot(snapshot.view_name.c_str(), snapshot.snapshot);
                snapshot.snapshot = permanent_snapshot;
                break;
            }
        }
    }
}

static bool IsUnqualifiedOrTemporarySchemaReference(const std::vector<std::string> &qualified_name);

static bool TableRefMatchesView(const std::vector<std::string> &table_name,
                                const std::string &view_name,
                                bool qualified_permanent) {
    if (table_name.empty() || !EqualsCaseInsensitive(table_name.back(), view_name)) {
        return false;
    }
    bool temporary_reference = IsUnqualifiedOrTemporarySchemaReference(table_name);
    return qualified_permanent ? !temporary_reference : temporary_reference;
}

static std::string SelectPortionForTableAnalysis(const std::string &sql) {
    std::string dollar_quote_end;
    bool in_single_quote = false;
    bool in_double_quote = false;
    bool in_line_comment = false;
    bool in_block_comment = false;
    idx_t depth = 0;

    for (idx_t i = 0; i < sql.size(); i++) {
        char c = sql[i];
        char next = i + 1 < sql.size() ? sql[i + 1] : '\0';

        if (in_line_comment) {
            if (c == '\n') {
                in_line_comment = false;
            }
            continue;
        }
        if (in_block_comment) {
            if (c == '*' && next == '/') {
                in_block_comment = false;
                i++;
            }
            continue;
        }
        if (!dollar_quote_end.empty()) {
            if (sql.compare(i, dollar_quote_end.size(), dollar_quote_end) == 0) {
                i += dollar_quote_end.size() - 1;
                dollar_quote_end.clear();
            }
            continue;
        }
        if (in_single_quote) {
            if (c == '\'' && next == '\'') {
                i++;
            } else if (c == '\'') {
                in_single_quote = false;
            }
            continue;
        }
        if (in_double_quote) {
            if (c == '"' && next == '"') {
                i++;
            } else if (c == '"') {
                in_double_quote = false;
            }
            continue;
        }

        if (c == '-' && next == '-') {
            in_line_comment = true;
            i++;
            continue;
        }
        if (c == '/' && next == '*') {
            in_block_comment = true;
            i++;
            continue;
        }
        if (c == '\'') {
            in_single_quote = true;
            continue;
        }
        if (c == '"') {
            in_double_quote = true;
            continue;
        }
        if (c == '$') {
            idx_t tag_end = i + 1;
            while (tag_end < sql.size() && sql[tag_end] != '$') {
                char tag_char = sql[tag_end];
                if (!std::isalnum(static_cast<unsigned char>(tag_char)) && tag_char != '_') {
                    break;
                }
                tag_end++;
            }
            if (tag_end < sql.size() && sql[tag_end] == '$') {
                dollar_quote_end = sql.substr(i, tag_end - i + 1);
                i = tag_end;
                continue;
            }
        }

        if (c == '(') {
            depth++;
            continue;
        }
        if (c == ')' && depth > 0) {
            depth--;
            continue;
        }

        size_t pos = i;
        if (depth == 0 && ConsumeKeyword(sql, pos, "WITH")) {
            return sql.substr(i);
        }

        pos = i;
        if (depth == 0 && ConsumeKeyword(sql, pos, "SELECT")) {
            return sql.substr(i);
        }
    }

    return sql;
}

static std::vector<std::string> ReadQualifiedIdentifier(const std::string &sql, size_t &pos) {
    std::vector<std::string> parts;
    while (pos < sql.size()) {
        std::string part;
        if (sql[pos] == '"') {
            pos++;
            bool closed = false;
            while (pos < sql.size()) {
                if (sql[pos] == '"' && pos + 1 < sql.size() && sql[pos + 1] == '"') {
                    part += '"';
                    pos += 2;
                    continue;
                }
                if (sql[pos] == '"') {
                    pos++;
                    closed = true;
                    break;
                }
                part += sql[pos++];
            }
            if (!closed || part.empty()) {
                return {};
            }
        } else {
            size_t start = pos;
            while (pos < sql.size() && IsIdentifierChar(sql[pos])) {
                pos++;
            }
            if (pos == start) {
                return {};
            }
            part = sql.substr(start, pos - start);
        }
        parts.push_back(std::move(part));
        pos = SkipWhitespaceAndComments(sql, pos);
        if (pos == sql.size() || sql[pos] != '.') {
            return parts;
        }
        pos = SkipWhitespaceAndComments(sql, pos + 1);
    }
    return {}; // A trailing dot does not form a qualified identifier.
}

static bool IsUnqualifiedOrTemporarySchemaReference(const std::vector<std::string> &qualified_name) {
    if (qualified_name.size() < 2) {
        return true;
    }

    auto &schema = qualified_name[qualified_name.size() - 2];
    return EqualsCaseInsensitive(schema, "temp") ||
           EqualsCaseInsensitive(schema, "temporary") ||
           EqualsCaseInsensitive(schema, "pg_temp");
}

static bool IsExtractFromKeyword(const std::string &sql, idx_t from_pos) {
    idx_t depth = 0;
    for (idx_t i = from_pos; i > 0; i--) {
        char c = sql[i - 1];
        if (c == ')') {
            depth++;
            continue;
        }
        if (c != '(') {
            continue;
        }
        if (depth > 0) {
            depth--;
            continue;
        }

        idx_t token_end = i - 1;
        while (token_end > 0 && std::isspace(static_cast<unsigned char>(sql[token_end - 1]))) {
            token_end--;
        }
        idx_t token_start = token_end;
        while (token_start > 0 && IsIdentifierChar(sql[token_start - 1])) {
            token_start--;
        }
        return EqualsCaseInsensitive(sql.substr(token_start, token_end - token_start), "EXTRACT");
    }
    return false;
}

static bool StatementTextReadsFromView(const std::string &sql,
                                       const std::string &view_name,
                                       bool qualified_permanent) {
    std::string dollar_quote_end;
    bool in_single_quote = false;
    bool in_double_quote = false;
    bool in_line_comment = false;
    bool in_block_comment = false;

    for (idx_t i = 0; i < sql.size(); i++) {
        char c = sql[i];
        char next = i + 1 < sql.size() ? sql[i + 1] : '\0';

        if (in_line_comment) {
            if (c == '\n') {
                in_line_comment = false;
            }
            continue;
        }
        if (in_block_comment) {
            if (c == '*' && next == '/') {
                in_block_comment = false;
                i++;
            }
            continue;
        }
        if (!dollar_quote_end.empty()) {
            if (sql.compare(i, dollar_quote_end.size(), dollar_quote_end) == 0) {
                i += dollar_quote_end.size() - 1;
                dollar_quote_end.clear();
            }
            continue;
        }
        if (in_single_quote) {
            if (c == '\'' && next == '\'') {
                i++;
            } else if (c == '\'') {
                in_single_quote = false;
            }
            continue;
        }
        if (in_double_quote) {
            if (c == '"' && next == '"') {
                i++;
            } else if (c == '"') {
                in_double_quote = false;
            }
            continue;
        }

        if (c == '-' && next == '-') {
            in_line_comment = true;
            i++;
            continue;
        }
        if (c == '/' && next == '*') {
            in_block_comment = true;
            i++;
            continue;
        }
        if (c == '\'') {
            in_single_quote = true;
            continue;
        }
        if (c == '"') {
            in_double_quote = true;
            continue;
        }
        if (c == '$') {
            idx_t tag_end = i + 1;
            while (tag_end < sql.size() && sql[tag_end] != '$') {
                char tag_char = sql[tag_end];
                if (!std::isalnum(static_cast<unsigned char>(tag_char)) && tag_char != '_') {
                    break;
                }
                tag_end++;
            }
            if (tag_end < sql.size() && sql[tag_end] == '$') {
                dollar_quote_end = sql.substr(i, tag_end - i + 1);
                i = tag_end;
                continue;
            }
        }

        size_t pos = i;
        bool from_keyword = ConsumeKeyword(sql, pos, "FROM");
        if (from_keyword && IsExtractFromKeyword(sql, i)) {
            continue;
        }
        if (!from_keyword && !ConsumeKeyword(sql, pos, "JOIN")) {
            continue;
        }

        pos = SkipWhitespaceAndComments(sql, pos);
        auto table_name = ReadQualifiedIdentifier(sql, pos);
        if (!table_name.empty() && TableRefMatchesView(table_name, view_name, qualified_permanent)) {
            return true;
        }
    }

    return false;
}

static bool StatementReadsFromView(const std::string &sql,
                                   const std::string &view_name,
                                   bool qualified_permanent = false) {
    std::string select_sql = SelectPortionForTableAnalysis(sql);
    YardstickSelectInfo *info = yardstick_parse_select(select_sql.c_str());
    if (!info || info->error) {
        if (info) {
            yardstick_free_select_info(info);
        }
        return StatementTextReadsFromView(select_sql, view_name, qualified_permanent);
    }

    bool found = false;
    bool has_subquery = false;
    for (idx_t i = 0; i < info->table_count; i++) {
        if (info->tables[i].is_subquery) {
            has_subquery = true;
            continue;
        }
        // Parser FFI exposes the decoded final name, without qualification.
        // Preserve literal dots here; qualified permanent references are
        // resolved by the source-aware fallback below.
        if (!qualified_permanent && info->tables[i].table_name &&
            EqualsCaseInsensitive(info->tables[i].table_name, view_name)) {
            found = true;
            break;
        }
    }

    size_t select_pos = SkipWhitespaceAndComments(select_sql, 0);
    bool starts_with_with = ConsumeKeyword(select_sql, select_pos, "WITH");
    yardstick_free_select_info(info);
    return found || ((qualified_permanent || starts_with_with || has_subquery) &&
                     StatementTextReadsFromView(select_sql, view_name, qualified_permanent));
}

struct DropViewInfo {
    std::string sql;
    std::string view_name;
    bool targets_temporary_view = true;
};

static bool ExtractDropViewInfoFromSql(const std::string &sql, DropViewInfo &drop_view) {
    size_t pos = SkipWhitespaceAndComments(sql, 0);
    if (!ConsumeKeyword(sql, pos, "DROP")) {
        return false;
    }
    pos = SkipWhitespaceAndComments(sql, pos);
    if (!ConsumeKeyword(sql, pos, "VIEW")) {
        return false;
    }
    pos = SkipWhitespaceAndComments(sql, pos);
    if (ConsumeKeyword(sql, pos, "IF")) {
        pos = SkipWhitespaceAndComments(sql, pos);
        if (!ConsumeKeyword(sql, pos, "EXISTS")) {
            return false;
        }
        pos = SkipWhitespaceAndComments(sql, pos);
    }

    auto qualified_name = ReadQualifiedIdentifier(sql, pos);
    if (qualified_name.empty()) {
        return false;
    }
    pos = SkipWhitespaceAndComments(sql, pos);
    if (ConsumeKeyword(sql, pos, "CASCADE") ||
        ConsumeKeyword(sql, pos, "RESTRICT")) {
        pos = SkipWhitespaceAndComments(sql, pos);
    }
    while (pos < sql.size() && sql[pos] == ';') {
        pos++;
        pos = SkipWhitespaceAndComments(sql, pos);
    }
    if (pos < sql.size()) {
        return false;
    }

    drop_view.sql = sql;
    drop_view.view_name = qualified_name.back();
    drop_view.targets_temporary_view = IsUnqualifiedOrTemporarySchemaReference(qualified_name);
    return true;
}

static bool SnapshotAndDropMeasureViewFromSql(const std::string &sql,
                                              const std::string &view_name,
                                              bool targets_temporary_view,
                                              std::vector<MeasureViewSnapshot> &temporary_snapshots,
                                              std::vector<std::string> &pending_temporary_views,
                                              std::vector<std::string> &used_temporary_views,
                                              std::vector<MeasureViewSnapshot> &snapshots) {
    if (targets_temporary_view && HasMeasureSnapshot(temporary_snapshots, view_name)) {
        RemoveCaseInsensitive(pending_temporary_views, view_name);
        RemoveCaseInsensitive(used_temporary_views, view_name);
        yardstick_drop_measure_view_from_sql(sql.c_str());
        RestoreAndRemoveTemporarySnapshot(temporary_snapshots, view_name);
        return true;
    }
    if (!targets_temporary_view &&
        TransferTemporarySnapshotForPermanentDrop(temporary_snapshots, view_name, snapshots)) {
        return true;
    }

    snapshots.push_back(SnapshotMeasureView(view_name));
    yardstick_drop_measure_view_from_sql(sql.c_str());
    return true;
}

static string EscapeSqlStringLiteral(const string &sql) {
    string escaped_sql;
    for (char c : sql) {
        if (c == '\'') {
            escaped_sql += "''";
        } else {
            escaped_sql += c;
        }
    }
    return escaped_sql;
}

static std::string AggregateWarnings(YardstickAggregateResult &result) {
    return result.warnings ? string(result.warnings) : string();
}

static bool WarningsAsErrors(ClientContext &context) {
    Value setting;
    if (!context.TryGetCurrentSetting("warnings_as_errors", setting) || setting.IsNull()) {
        return false;
    }
    return setting.GetValue<bool>();
}

static void HandleAggregateWarnings(ClientContext &context, const string &warnings) {
    if (warnings.empty()) {
        return;
    }
    if (WarningsAsErrors(context)) {
        throw InvalidInputException("%s", warnings);
    }
    DUCKDB_LOG_WARNING(context, "%s", warnings.c_str());
}

static string YardstickWrapperSql(const string &expanded_sql, const string &warnings) {
    string wrapper_sql = "SELECT * FROM yardstick('" + EscapeSqlStringLiteral(expanded_sql) + "'";
    if (!warnings.empty()) {
        wrapper_sql += ", '" + EscapeSqlStringLiteral(warnings) + "'";
    }
    wrapper_sql += ")";
    return wrapper_sql;
}

static string WrapYardstickSelect(const string &sql) {
    return YardstickWrapperSql(sql, "");
}

static string WarningStatementSql(const string &warnings) {
    return "SELECT yardstick_warning('" + EscapeSqlStringLiteral(warnings) + "')";
}

static bool ParsesAsSingleSelect(const string &sql) {
    Parser parser(YardstickParserOptions());
    try {
        parser.ParseQuery(sql);
    } catch (...) {
        return false;
    }
    return parser.statements.size() == 1 &&
           parser.statements[0]->type == StatementType::SELECT_STATEMENT;
}

static bool IsKeywordBoundary(char c) {
    return !std::isalnum(static_cast<unsigned char>(c)) && c != '_';
}

static idx_t FindTopLevelKeyword(const string &sql, const string &keyword, idx_t start = 0) {
    idx_t depth = 0;
    for (idx_t i = start; i < sql.size(); i++) {
        char c = sql[i];
        if (c == '\'') {
            i++;
            while (i < sql.size()) {
                if (sql[i] == '\'' && i + 1 < sql.size() && sql[i + 1] == '\'') {
                    i += 2;
                    continue;
                }
                if (sql[i] == '\'') {
                    break;
                }
                i++;
            }
            continue;
        }
        if (c == '"') {
            i++;
            while (i < sql.size()) {
                if (sql[i] == '"' && i + 1 < sql.size() && sql[i + 1] == '"') {
                    i += 2;
                    continue;
                }
                if (sql[i] == '"') {
                    break;
                }
                i++;
            }
            continue;
        }
        if (c == '-' && i + 1 < sql.size() && sql[i + 1] == '-') {
            i += 2;
            while (i < sql.size() && sql[i] != '\n' && sql[i] != '\r') {
                i++;
            }
            continue;
        }
        if (c == '/' && i + 1 < sql.size() && sql[i + 1] == '*') {
            i += 2;
            while (i + 1 < sql.size() && !(sql[i] == '*' && sql[i + 1] == '/')) {
                i++;
            }
            if (i + 1 < sql.size()) {
                i++;
            }
            continue;
        }
        if (c == '(') {
            depth++;
            continue;
        }
        if (c == ')' && depth > 0) {
            depth--;
            continue;
        }
        if (depth != 0 || i + keyword.size() > sql.size()) {
            continue;
        }
        bool matched = true;
        for (idx_t j = 0; j < keyword.size(); j++) {
            if (std::toupper(static_cast<unsigned char>(sql[i + j])) != keyword[j]) {
                matched = false;
                break;
            }
        }
        if (!matched) {
            continue;
        }
        bool left_ok = i == 0 || IsKeywordBoundary(sql[i - 1]);
        bool right_ok = i + keyword.size() >= sql.size() || IsKeywordBoundary(sql[i + keyword.size()]);
        if (left_ok && right_ok) {
            return i;
        }
    }
    return std::string::npos;
}

static idx_t SkipSqlWhitespace(const string &sql, idx_t i) {
    while (i < sql.size() && std::isspace(static_cast<unsigned char>(sql[i]))) {
        i++;
    }
    return i;
}

static idx_t SkipSqlWhitespaceAndComments(const string &sql, idx_t i) {
    while (i < sql.size()) {
        i = SkipSqlWhitespace(sql, i);
        if (i + 1 < sql.size() && sql[i] == '-' && sql[i + 1] == '-') {
            i += 2;
            while (i < sql.size() && sql[i] != '\n' && sql[i] != '\r') {
                i++;
            }
            continue;
        }
        if (i + 1 < sql.size() && sql[i] == '/' && sql[i + 1] == '*') {
            i += 2;
            while (i + 1 < sql.size() && !(sql[i] == '*' && sql[i + 1] == '/')) {
                i++;
            }
            if (i + 1 < sql.size()) {
                i += 2;
            }
            continue;
        }
        return i;
    }
    return i;
}

static bool KeywordAt(const string &sql, idx_t pos, const string &keyword) {
    if (pos + keyword.size() > sql.size()) {
        return false;
    }
    for (idx_t i = 0; i < keyword.size(); i++) {
        if (std::toupper(static_cast<unsigned char>(sql[pos + i])) != keyword[i]) {
            return false;
        }
    }
    bool left_ok = pos == 0 || IsKeywordBoundary(sql[pos - 1]);
    bool right_ok = pos + keyword.size() >= sql.size() || IsKeywordBoundary(sql[pos + keyword.size()]);
    return left_ok && right_ok;
}

static string StripTrailingSemicolon(string sql) {
    StringUtil::RTrim(sql);
    if (!sql.empty() && sql.back() == ';') {
        sql.pop_back();
        StringUtil::RTrim(sql);
    }
    return sql;
}

static idx_t FindMatchingParen(const string &sql, idx_t open_pos) {
    if (open_pos >= sql.size() || sql[open_pos] != '(') {
        return std::string::npos;
    }
    idx_t depth = 0;
    for (idx_t i = open_pos; i < sql.size(); i++) {
        char c = sql[i];
        if (c == '\'') {
            i++;
            while (i < sql.size()) {
                if (sql[i] == '\'' && i + 1 < sql.size() && sql[i + 1] == '\'') {
                    i += 2;
                    continue;
                }
                if (sql[i] == '\'') {
                    break;
                }
                i++;
            }
            continue;
        }
        if (c == '"') {
            i++;
            while (i < sql.size()) {
                if (sql[i] == '"' && i + 1 < sql.size() && sql[i + 1] == '"') {
                    i += 2;
                    continue;
                }
                if (sql[i] == '"') {
                    break;
                }
                i++;
            }
            continue;
        }
        if (c == '-' && i + 1 < sql.size() && sql[i + 1] == '-') {
            i += 2;
            while (i < sql.size() && sql[i] != '\n' && sql[i] != '\r') {
                i++;
            }
            continue;
        }
        if (c == '/' && i + 1 < sql.size() && sql[i + 1] == '*') {
            i += 2;
            while (i + 1 < sql.size() && !(sql[i] == '*' && sql[i + 1] == '/')) {
                i++;
            }
            if (i + 1 < sql.size()) {
                i++;
            }
            continue;
        }
        if (c == '(') {
            depth++;
        } else if (c == ')') {
            depth--;
            if (depth == 0) {
                return i;
            }
        }
    }
    return std::string::npos;
}

static bool IsWrappedInParens(const string &sql) {
    idx_t start = SkipSqlWhitespace(sql, 0);
    if (start >= sql.size() || sql[start] != '(') {
        return false;
    }
    idx_t close = FindMatchingParen(sql, start);
    if (close == std::string::npos) {
        return false;
    }
    idx_t tail = SkipSqlWhitespace(sql, close + 1);
    return tail == sql.size() || sql[tail] == ';';
}

static string UnwrapQueryBody(string sql) {
    sql = StripTrailingSemicolon(sql);
    while (IsWrappedInParens(sql)) {
        idx_t start = SkipSqlWhitespace(sql, 0);
        idx_t close = FindMatchingParen(sql, start);
        sql = sql.substr(start + 1, close - start - 1);
        sql = StripTrailingSemicolon(sql);
    }
    return sql;
}

static string WarningWrappedSelectSql(const string &query_sql, const string &warnings) {
    string query = UnwrapQueryBody(query_sql);
    return "WITH __yardstick_warning AS MATERIALIZED (" + WarningStatementSql(warnings) +
           "), __yardstick_query AS MATERIALIZED (\n" + query +
           "\n) SELECT __yardstick_query.* FROM __yardstick_warning CROSS JOIN __yardstick_query";
}

static idx_t FindParenthesizedQueryBodyAfter(const string &sql, idx_t start) {
    idx_t i = start;
    while (i < sql.size()) {
        i = SkipSqlWhitespace(sql, i);
        if (i >= sql.size()) {
            return std::string::npos;
        }
        if (sql[i] == '(') {
            auto close = FindMatchingParen(sql, i);
            if (close == std::string::npos) {
                return std::string::npos;
            }
            auto body_start = SkipSqlWhitespaceAndComments(sql, i + 1);
            if (KeywordAt(sql, body_start, "SELECT") || KeywordAt(sql, body_start, "WITH")) {
                return i;
            }
            i = close + 1;
            continue;
        }
        i++;
    }
    return std::string::npos;
}

static bool TryWarningWrappedNonSelectSql(const string &expanded_sql,
                                          const string &warnings,
                                          string &rewritten_sql) {
    string trimmed = expanded_sql;
    StringUtil::LTrim(trimmed);
    string upper = StringUtil::Upper(trimmed);

    auto insert_pos = FindTopLevelKeyword(expanded_sql, "INSERT");
    if (insert_pos != std::string::npos) {
        auto select_pos = FindTopLevelKeyword(expanded_sql, "SELECT", insert_pos + strlen("INSERT"));
        auto with_pos = FindTopLevelKeyword(expanded_sql, "WITH", insert_pos + strlen("INSERT"));
        idx_t body_pos = std::string::npos;
        if (select_pos != std::string::npos && with_pos != std::string::npos) {
            body_pos = select_pos < with_pos ? select_pos : with_pos;
        } else if (select_pos != std::string::npos) {
            body_pos = select_pos;
        } else {
            body_pos = with_pos;
        }
        if (body_pos != std::string::npos) {
            rewritten_sql = expanded_sql.substr(0, body_pos) +
                            WarningWrappedSelectSql(expanded_sql.substr(body_pos), warnings);
            return true;
        }
        body_pos = FindParenthesizedQueryBodyAfter(expanded_sql, insert_pos + strlen("INSERT"));
        if (body_pos != std::string::npos) {
            rewritten_sql = expanded_sql.substr(0, body_pos) +
                            WarningWrappedSelectSql(expanded_sql.substr(body_pos), warnings);
            return true;
        }
    }

    auto create_pos = FindTopLevelKeyword(expanded_sql, "CREATE");
    auto table_pos = create_pos == std::string::npos
                         ? std::string::npos
                         : FindTopLevelKeyword(expanded_sql, "TABLE", create_pos + strlen("CREATE"));
    auto as_pos = table_pos == std::string::npos
                      ? std::string::npos
                      : FindTopLevelKeyword(expanded_sql, "AS", table_pos + strlen("TABLE"));
    if (as_pos != std::string::npos && !StringUtil::StartsWith(upper, "CREATE VIEW")) {
        auto body_pos = SkipSqlWhitespace(expanded_sql, as_pos + strlen("AS"));
        if (body_pos < expanded_sql.size()) {
            rewritten_sql = expanded_sql.substr(0, as_pos + strlen("AS")) + " " +
                            WarningWrappedSelectSql(expanded_sql.substr(body_pos), warnings);
            return true;
        }
    }

    return false;
}

static MeasureRewriteResult RewriteMeasureViewsStatementByStatement(
        const std::string &query,
        std::vector<MeasureViewSnapshot> &permanent_snapshots) {
    MeasureRewriteResult rewrite_result;
    std::vector<std::string> rewritten_statements;
    std::vector<MeasureViewSnapshot> temporary_snapshots;
    std::vector<std::string> pending_temporary_views;
    std::vector<std::string> used_temporary_views;
    std::vector<DropViewInfo> deferred_drop_views;
    bool catalog_statement_seen = false;
    bool executable_since_catalog_mutation = false;
    auto cleanup_temporary_measure_views = [&temporary_snapshots]() {
        RestoreMeasureViewSnapshots(temporary_snapshots);
    };
    auto apply_deferred_drop_views = [&](const std::string &skip_view_name = "") {
        for (auto &drop_view : deferred_drop_views) {
            if (!skip_view_name.empty() &&
                EqualsCaseInsensitive(drop_view.view_name, skip_view_name)) {
                continue;
            }
            SnapshotAndDropMeasureViewFromSql(drop_view.sql, drop_view.view_name,
                                              drop_view.targets_temporary_view,
                                              temporary_snapshots, pending_temporary_views,
                                              used_temporary_views,
                                              permanent_snapshots);
        }
        deferred_drop_views.clear();
    };
    auto has_deferred_drop_view = [&](const std::string &view_name) {
        for (auto &drop_view : deferred_drop_views) {
            if (EqualsCaseInsensitive(drop_view.view_name, view_name)) {
                return true;
            }
        }
        return false;
    };

    for (auto statement : SplitSqlStatements(query)) {
        StringUtil::Trim(statement);
        if (statement.empty()) {
            continue;
        }

        auto statement_body = statement.substr(SkipWhitespaceAndComments(statement, 0));
        std::string semantic_statement_body;
        if (StartsWithSemantic(statement_body, semantic_statement_body)) {
            statement_body = semantic_statement_body;
        }
        DropViewInfo drop_view;
        bool starts_with_drop_view = StartsWithDropViewStatement(statement_body);
        if (starts_with_drop_view && catalog_statement_seen && executable_since_catalog_mutation) {
            rewrite_result.error = "AS MEASURE batches cannot apply catalog changes after executable statements";
            cleanup_temporary_measure_views();
            return rewrite_result;
        }
        if (ExtractDropViewInfoFromSql(statement_body, drop_view)) {
            if (catalog_statement_seen && executable_since_catalog_mutation) {
                rewrite_result.error = "AS MEASURE batches cannot apply catalog changes after executable statements";
                cleanup_temporary_measure_views();
                return rewrite_result;
            }
            catalog_statement_seen = true;
            if (rewrite_result.had_measure_view) {
                SnapshotAndDropMeasureViewFromSql(statement_body, drop_view.view_name,
                                                  drop_view.targets_temporary_view,
                                                  temporary_snapshots, pending_temporary_views,
                                                  used_temporary_views,
                                                  permanent_snapshots);
            } else {
                deferred_drop_views.push_back(drop_view);
            }
            executable_since_catalog_mutation = false;
            rewritten_statements.push_back(statement_body);
            continue;
        }

        bool statement_has_measure = yardstick_has_as_measure(statement_body.c_str());
        std::string native_view_name;
#if YARDSTICK_GRAMMAR_EXTENSION
        if (auto *native = FindNativeYardstickMeasures(statement_body.c_str())) {
            statement_has_measure = native->is_measure_view;
            if (native->view_name) {
                native_view_name = native->view_name;
            }
            if (native->error) {
                rewrite_result.error = native->error;
            }
            yardstick_free_create_view_info(native);
            if (!rewrite_result.error.empty()) {
                cleanup_temporary_measure_views();
                return rewrite_result;
            }
        }
#endif
        if (statement_has_measure && StartsWithCreateViewStatement(statement_body)) {
            if (catalog_statement_seen && executable_since_catalog_mutation) {
                rewrite_result.error = "AS MEASURE batches cannot apply catalog changes after executable statements";
                cleanup_temporary_measure_views();
                return rewrite_result;
            }
            catalog_statement_seen = true;

            std::string rewritten_statement = RewritePercentileWithinGroup(statement_body);
            bool is_temporary_measure_view = IsTemporaryCreateViewStatement(statement_body);
            std::string view_name = native_view_name;
            if (view_name.empty()) {
                char *extracted_view_name = yardstick_extract_view_name(rewritten_statement.c_str());
                if (extracted_view_name) {
                    view_name = extracted_view_name;
                    yardstick_free(extracted_view_name);
                }
            }
            MeasureViewSnapshot snapshot;
            bool has_snapshot = false;
            if (!view_name.empty()) {
                snapshot = SnapshotMeasureView(view_name);
                has_snapshot = true;
            }
            bool preapplied_deferred_drop = false;
            if (is_temporary_measure_view && !view_name.empty() &&
                has_deferred_drop_view(view_name)) {
                apply_deferred_drop_views();
                if (has_snapshot) {
                    yardstick_free_measure_view_snapshot(snapshot.snapshot);
                    snapshot = SnapshotMeasureView(view_name);
                }
                preapplied_deferred_drop = true;
            }
            YardstickCreateViewResult result = yardstick_process_create_view(rewritten_statement.c_str());

            if (result.error) {
                rewrite_result.error = result.error;
                yardstick_free_create_view_result(result);
                if (has_snapshot) {
                    yardstick_free_measure_view_snapshot(snapshot.snapshot);
                }
                cleanup_temporary_measure_views();
                return rewrite_result;
            }

            if (result.is_measure_view) {
                rewrite_result.had_measure_view = true;
                executable_since_catalog_mutation = false;
                if (view_name.empty() && result.view_name) {
                    view_name = result.view_name;
                }
                if (!preapplied_deferred_drop) {
                    apply_deferred_drop_views(view_name);
                }
                if (is_temporary_measure_view && !view_name.empty()) {
                    if (has_snapshot) {
                        temporary_snapshots.push_back(snapshot);
                        pending_temporary_views.push_back(view_name);
                        has_snapshot = false;
                    }
                } else if (has_snapshot) {
                    permanent_snapshots.push_back(snapshot);
                    has_snapshot = false;
                }
                rewritten_statements.push_back(result.clean_sql);
            } else {
                rewritten_statements.push_back(statement);
            }

            if (has_snapshot) {
                yardstick_free_measure_view_snapshot(snapshot.snapshot);
            }
            yardstick_free_create_view_result(result);
            continue;
        }

        std::string aggregate_statement = statement_body;
        std::string semantic_stripped;
        if (StartsWithSemantic(aggregate_statement, semantic_stripped)) {
            aggregate_statement = semantic_stripped;
        }

        if (!yardstick_has_aggregate(aggregate_statement.c_str())) {
            rewritten_statements.push_back(statement);
            if (catalog_statement_seen) {
                executable_since_catalog_mutation = true;
            }
            continue;
        }

        std::vector<std::string> read_temporary_views;
        for (auto &view_name : pending_temporary_views) {
            if (StatementReadsFromView(aggregate_statement, view_name)) {
                read_temporary_views.push_back(view_name);
            }
        }
        if (!read_temporary_views.empty() && StartsWithSelectStatement(aggregate_statement)) {
            rewrite_result.error = "TEMPORARY AS MEASURE views cannot be returned directly from a statement batch";
            cleanup_temporary_measure_views();
            return rewrite_result;
        }

        std::vector<std::string> restored_shadowed_permanent_views;
        for (auto &view_name : pending_temporary_views) {
            if (StatementReadsFromView(aggregate_statement, view_name, true) &&
                RestoreShadowedPermanentMetadata(temporary_snapshots, view_name)) {
                restored_shadowed_permanent_views.push_back(view_name);
            }
        }

        YardstickAggregateResult result = yardstick_expand_aggregate(aggregate_statement.c_str());
        if (result.error) {
            RestoreTemporaryMetadata(temporary_snapshots, restored_shadowed_permanent_views);
            yardstick_free_aggregate_result(result);
            rewritten_statements.push_back(statement);
            if (catalog_statement_seen) {
                executable_since_catalog_mutation = true;
            }
            continue;
        }

        if (result.had_aggregate) {
            rewrite_result.had_aggregate = true;
            bool reads_pending_temporary_view = !read_temporary_views.empty();
            for (auto &view_name : read_temporary_views) {
                if (!ContainsCaseInsensitive(used_temporary_views, view_name)) {
                    used_temporary_views.push_back(view_name);
                }
            }
            string expanded_sql(result.expanded_sql);
            string warnings = AggregateWarnings(result);
            if (ParsesAsSingleSelect(expanded_sql)) {
                if (reads_pending_temporary_view) {
                    rewrite_result.error = "TEMPORARY AS MEASURE views cannot be returned directly from a statement batch";
                    yardstick_free_aggregate_result(result);
                    RestoreTemporaryMetadata(temporary_snapshots, restored_shadowed_permanent_views);
                    cleanup_temporary_measure_views();
                    return rewrite_result;
                }
                rewritten_statements.push_back(YardstickWrapperSql(expanded_sql, warnings));
            } else {
                if (!warnings.empty()) {
                    string warning_sql;
                    if (TryWarningWrappedNonSelectSql(expanded_sql, warnings, warning_sql)) {
                        rewritten_statements.push_back(warning_sql);
                    } else {
                        rewritten_statements.push_back(WarningStatementSql(warnings) + ";\n" + expanded_sql);
                    }
                } else {
                    rewritten_statements.push_back(expanded_sql);
                }
            }
        } else {
            rewritten_statements.push_back(statement);
        }
        RestoreTemporaryMetadata(temporary_snapshots, restored_shadowed_permanent_views);
        if (catalog_statement_seen) {
            executable_since_catalog_mutation = true;
        }
        yardstick_free_aggregate_result(result);
    }

    for (auto &view_name : pending_temporary_views) {
        if (ContainsCaseInsensitive(used_temporary_views, view_name)) {
            continue;
        }
        rewrite_result.error = "TEMPORARY AS MEASURE views must be used in the same statement batch as AGGREGATE()";
        cleanup_temporary_measure_views();
        return rewrite_result;
    }

    for (idx_t i = 0; i < rewritten_statements.size(); i++) {
        if (i > 0) {
            rewrite_result.rewritten_sql += ";\n";
        }
        rewrite_result.rewritten_sql += rewritten_statements[i];
    }

    cleanup_temporary_measure_views();
    return rewrite_result;
}

#if YARDSTICK_TOKEN_PARSE_FN
// DuckDB main: parse_function receives the post-PEG-failure token tail rather
// than the query string. Yardstick rewrites queries earlier, in
// yardstick_parser_override (which sees the full query string), so there is
// nothing to do here -- decline and let DuckDB proceed.
ParserExtensionParseResult yardstick_parse(ParserExtensionInfo *,
                                            const vector<SimpleToken> &) {
    return ParserExtensionParseResult();
}
#else
ParserExtensionParseResult yardstick_parse(ParserExtensionInfo *,
                                            const std::string &query) {

    // Determine the SQL to check (strip SEMANTIC prefix if present)
    std::string sql_to_check = query;
    std::string semantic_stripped;
    bool had_semantic_prefix = StartsWithSemantic(query, semantic_stripped);
    if (had_semantic_prefix) {
        sql_to_check = semantic_stripped;
    }

    if (yardstick_drop_measure_view_from_sql(sql_to_check.c_str())) {
        return ParserExtensionParseResult();
    }

    bool had_measure_rewrite = false;
    std::vector<MeasureViewSnapshot> permanent_snapshots;
    if (yardstick_has_as_measure(sql_to_check.c_str())) {
        auto measure_rewrite = RewriteMeasureViewsStatementByStatement(sql_to_check, permanent_snapshots);
        if (!measure_rewrite.error.empty()) {
            RestoreMeasureViewSnapshots(permanent_snapshots);
            return ParserExtensionParseResult(measure_rewrite.error);
        }
        if (measure_rewrite.had_measure_view) {
            had_measure_rewrite = true;
            sql_to_check = std::move(measure_rewrite.rewritten_sql);
        }
    }

    // Check for AGGREGATE() function
    if (!had_measure_rewrite && yardstick_has_aggregate(sql_to_check.c_str())) {
        YardstickAggregateResult result = yardstick_expand_aggregate(sql_to_check.c_str());

        if (result.error) {
            string error_msg(result.error);
            yardstick_free_aggregate_result(result);
            RestoreMeasureViewSnapshots(permanent_snapshots);
            return ParserExtensionParseResult(error_msg);
        }

        if (result.had_aggregate) {
            string expanded_sql(result.expanded_sql);
            string warnings = AggregateWarnings(result);
            yardstick_free_aggregate_result(result);

            // Wrap in table function call
            string wrapper_sql = YardstickWrapperSql(expanded_sql, warnings);

            Parser parser(YardstickParserOptions());
            parser.ParseQuery(wrapper_sql);
            auto statements = std::move(parser.statements);

            if (statements.empty()) {
                RestoreMeasureViewSnapshots(permanent_snapshots);
                return ParserExtensionParseResult("Table function wrapper produced no statements");
            }

            RestoreMeasureViewSnapshots(permanent_snapshots);
            return ParserExtensionParseResult(
                make_uniq_base<ParserExtensionParseData, YardstickParseData>(
                    std::move(statements[0])));
        }

        yardstick_free_aggregate_result(result);
    }

    if (had_measure_rewrite) {
        Parser parser(YardstickParserOptions());
        try {
            parser.ParseQuery(sql_to_check);
        } catch (std::exception &e) {
            RestoreMeasureViewSnapshots(permanent_snapshots);
            return ParserExtensionParseResult(e.what());
        }
        auto statements = std::move(parser.statements);

        if (statements.empty()) {
            RestoreMeasureViewSnapshots(permanent_snapshots);
            return ParserExtensionParseResult("CREATE VIEW produced no statements");
        }

        FreeMeasureViewSnapshots(permanent_snapshots);
        return ParserExtensionParseResult(
            make_uniq_base<ParserExtensionParseData, YardstickParseData>(
                std::move(statements[0])));
    }

    RestoreMeasureViewSnapshots(permanent_snapshots);

    // Not a yardstick query, let DuckDB handle it
    return ParserExtensionParseResult();
}
#endif // YARDSTICK_TOKEN_PARSE_FN

//=============================================================================
// PARSER OVERRIDE: intercepts ALL queries before DuckDB's native parser
//=============================================================================

#if YARDSTICK_GRAMMAR_EXTENSION
static std::atomic<size_t> deferred_temporary_contexts {0};

static vector<unique_ptr<SQLStatement>> DeferMeasureColumnListBatch(const string &query) {
    if (CurrentNativeYardstickClientContext()) {
        return {};
    }
    bool requires_binding = deferred_temporary_contexts.load() != 0;
    auto statements = SplitSqlStatements(query);
    for (auto &sql : statements) {
        string semantic_stripped;
        if (StartsWithSemantic(sql, semantic_stripped)) {
            sql = std::move(semantic_stripped);
        }
        auto inspect_calls = [&](const string &scope_sql) {
            bool inspect_nested_scopes = false;
            if (auto *calls = FindNativeYardstickAggregates(scope_sql.c_str())) {
                for (size_t index = 0; index < calls->count; index++) {
                    // Decorations need the originating catalog for aggregate
                    // classification and star expansion before adding lineage.
                    requires_binding |= calls->calls[index].has_decorations;
                    inspect_nested_scopes |= calls->calls[index].modifier_count != 0;
                }
                yardstick_free_aggregate_list(calls);
            } else {
                inspect_nested_scopes = true;
            }
            return inspect_nested_scopes;
        };
        if (inspect_calls(sql) && !requires_binding) {
            // AT modifier expressions can own nested queries that are absent
            // from the enclosing call's marker AST. Inspect their native spans.
            if (auto *scopes = FindNativeYardstickQueryScopes(sql.c_str())) {
                for (size_t index = 0; index < scopes->count && !requires_binding; index++) {
                    auto &scope = scopes->scopes[index];
                    inspect_calls(sql.substr(scope.start_pos, scope.end_pos - scope.start_pos));
                }
                yardstick_free_query_scopes(scopes);
            }
        }
        if (!StartsWithCreateViewStatement(sql)) {
            continue;
        }
        if (auto *native = FindNativeYardstickMeasures(sql.c_str())) {
            requires_binding |= native->requires_binding;
            yardstick_free_create_view_info(native);
        }
    }
    if (!requires_binding) {
        return {};
    }
    vector<unique_ptr<SQLStatement>> deferred;
    auto options = *CurrentNativeYardstickParserOptions();
    for (auto &sql : statements) {
        StringUtil::Trim(sql);
        if (sql.empty()) {
            continue;
        }
        // Transaction and setting statements must retain their engine-visible
        // types so statement preprocessing preserves session semantics.
        Parser parser(options);
        if (!ParseNativeYardstickQuery(sql, parser)) {
            return {};
        }
        if (parser.statements.size() == 1 &&
            (parser.statements[0]->type == StatementType::TRANSACTION_STATEMENT ||
             parser.statements[0]->type == StatementType::SET_STATEMENT ||
             parser.statements[0]->type == StatementType::PRAGMA_STATEMENT ||
             parser.statements[0]->type == StatementType::EXECUTE_STATEMENT)) {
            deferred.push_back(std::move(parser.statements[0]));
        } else if (parser.statements.size() == 1 &&
                   parser.statements[0]->type == StatementType::SELECT_STATEMENT) {
            Parser wrapper_parser(options);
            wrapper_parser.ParseQuery("SELECT * FROM yardstick_scoped('" + EscapeSqlStringLiteral(sql) + "')");
            auto statement = std::move(wrapper_parser.statements[0]);
            statement->named_param_map = parser.statements[0]->named_param_map;
            statement->has_anonymous_parameters = parser.statements[0]->has_anonymous_parameters;
            deferred.push_back(std::move(statement));
        } else {
            auto statement = make_uniq<ExtensionStatement>(YardstickParserExtension(),
                make_uniq_base<ParserExtensionParseData, YardstickDeferredParseData>(sql, options));
            if (parser.statements.size() == 1) {
                statement->named_param_map = parser.statements[0]->named_param_map;
                statement->has_anonymous_parameters = parser.statements[0]->has_anonymous_parameters;
            }
            deferred.push_back(std::move(statement));
        }
    }
    return deferred;
}
#endif

ParserOverrideResult yardstick_parser_override(ParserExtensionInfo *info,
                                                const std::string &query,
                                                ParserOptions &options) {
#if YARDSTICK_GRAMMAR_EXTENSION
    NativeYardstickParseScope native_scope(info, options);
#endif
    // Strip SEMANTIC prefix if present (backwards compatibility)
    std::string sql_to_check = query;
    std::string semantic_stripped;
    bool had_semantic_prefix = StartsWithSemantic(query, semantic_stripped);
    if (had_semantic_prefix) {
        sql_to_check = semantic_stripped;
    }

#if YARDSTICK_GRAMMAR_EXTENSION
    try {
        auto deferred = DeferMeasureColumnListBatch(sql_to_check);
        if (!deferred.empty()) {
            return ParserOverrideResult(std::move(deferred));
        }
    } catch (std::exception &error) {
        return ParserOverrideResult(error);
    }
#endif

    bool native_has_measure = false;
    bool native_parsed = false;
#if YARDSTICK_GRAMMAR_EXTENSION
    native_parsed = NormalizeYardstickGrammar(info, options, sql_to_check, native_has_measure);
#endif

    // Check for DROP VIEW on measure views
    if (yardstick_drop_measure_view_from_sql(sql_to_check.c_str())) {
        // Catalog cleanup done; let DuckDB handle the actual DROP
        return ParserOverrideResult();
    }

    bool had_measure_rewrite = false;
    std::vector<MeasureViewSnapshot> permanent_snapshots;
    if (native_parsed ? native_has_measure : yardstick_has_as_measure(sql_to_check.c_str())) {
        auto measure_rewrite = RewriteMeasureViewsStatementByStatement(sql_to_check, permanent_snapshots);
        if (!measure_rewrite.error.empty()) {
            RestoreMeasureViewSnapshots(permanent_snapshots);
            try {
                throw ParserException(measure_rewrite.error);
            } catch (std::exception &e) {
                return ParserOverrideResult(e);
            }
        }
        if (measure_rewrite.had_measure_view) {
            had_measure_rewrite = true;
            sql_to_check = std::move(measure_rewrite.rewritten_sql);
        }
    }

    // Check for AGGREGATE() function
    if (!had_measure_rewrite && yardstick_has_aggregate(sql_to_check.c_str())) {
        YardstickAggregateResult result = yardstick_expand_aggregate(sql_to_check.c_str());

        if (result.error) {
            if (native_parsed && result.had_aggregate) {
                // Native discovery distinguishes measure calls from DuckDB's
                // list aggregate. Retrying a failed semantic rewrite through
                // compatibility lowering can silently lose query correlations.
                string error_msg(result.error);
                yardstick_free_aggregate_result(result);
                RestoreMeasureViewSnapshots(permanent_snapshots);
                // DISPLAY_EXTENSION_ERROR is ignored in DuckDB's fallback
                // override mode. This is a recognized semantic failure, so it
                // must propagate instead of inviting another parser attempt.
                throw ParserException(error_msg);
            }
            // Expansion failed: this might not be a yardstick AGGREGATE() call
            // (e.g. DuckDB's built-in list aggregate function). Fall through to
            // the native parser in case it can handle the query.
            yardstick_free_aggregate_result(result);
            RestoreMeasureViewSnapshots(permanent_snapshots);
            return ParserOverrideResult();
        }

        if (result.had_aggregate) {
            string expanded_sql(result.expanded_sql);
            string warnings = AggregateWarnings(result);
            yardstick_free_aggregate_result(result);

            // Validate the expanded SQL parses. If expansion produced garbage
            // (e.g. because AGGREGATE() was actually DuckDB's list aggregate
            // function, not a yardstick measure), fall through to the native parser.
            Parser validation_parser(YardstickParserOptions());
            try {
                validation_parser.ParseQuery(expanded_sql);
            } catch (...) {
                RestoreMeasureViewSnapshots(permanent_snapshots);
                return ParserOverrideResult();
            }

            // For SELECT statements, wrap in yardstick() table function so that
            // any remaining AGGREGATE() calls get a second expansion pass.
            // For non-SELECT (CTAS, INSERT...SELECT), return parsed statements
            // directly to preserve the caller's transaction context.
            bool is_select = !validation_parser.statements.empty() &&
                             validation_parser.statements[0]->type == StatementType::SELECT_STATEMENT;

            if (is_select) {
                string wrapper_sql = YardstickWrapperSql(expanded_sql, warnings);
                Parser parser(YardstickParserOptions());
                parser.ParseQuery(wrapper_sql);
                RestoreMeasureViewSnapshots(permanent_snapshots);
                return ParserOverrideResult(std::move(parser.statements));
            }

            if (!warnings.empty()) {
                string warning_sql;
                if (TryWarningWrappedNonSelectSql(expanded_sql, warnings, warning_sql)) {
                    Parser parser(YardstickParserOptions());
                    parser.ParseQuery(warning_sql);
                    RestoreMeasureViewSnapshots(permanent_snapshots);
                    return ParserOverrideResult(std::move(parser.statements));
                }
                Parser parser(YardstickParserOptions());
                parser.ParseQuery(WarningStatementSql(warnings) + "; " + expanded_sql);
                RestoreMeasureViewSnapshots(permanent_snapshots);
                return ParserOverrideResult(std::move(parser.statements));
            }

            RestoreMeasureViewSnapshots(permanent_snapshots);
            return ParserOverrideResult(std::move(validation_parser.statements));
        }

        yardstick_free_aggregate_result(result);
    }

    if (had_measure_rewrite) {
        try {
            Parser parser(YardstickParserOptions());
            parser.ParseQuery(sql_to_check);
            FreeMeasureViewSnapshots(permanent_snapshots);
            return ParserOverrideResult(std::move(parser.statements));
        } catch (std::exception &e) {
            RestoreMeasureViewSnapshots(permanent_snapshots);
            return ParserOverrideResult(e);
        }
    }

    RestoreMeasureViewSnapshots(permanent_snapshots);

    // Not a yardstick query; fall through to DuckDB's native parser
    return ParserOverrideResult();
}

#if YARDSTICK_GRAMMAR_EXTENSION
class DeferredMeasureCatalogState : public ClientContextState {
public:
    using TemporaryViews = std::map<string, std::shared_ptr<void>>;
    TemporaryViews temporary_views;
    std::unique_ptr<TemporaryViews> pending_temporary;
    std::unique_ptr<TemporaryViews> transaction_temporary;
    bool counted = false;
    std::vector<MeasureViewSnapshot> pending;
    std::vector<MeasureViewSnapshot> transaction_snapshots;

    ~DeferredMeasureCatalogState() override {
        if (counted) {
            deferred_temporary_contexts.fetch_sub(1);
        }
        FreeMeasureViewSnapshots(pending);
        FreeMeasureViewSnapshots(transaction_snapshots);
    }
    void BeginTemporaryChange() {
        if (!pending_temporary) {
            pending_temporary = std::make_unique<TemporaryViews>(temporary_views);
        }
        if (!transaction_temporary) {
            transaction_temporary = std::make_unique<TemporaryViews>(temporary_views);
        }
        if (!counted) {
            counted = true;
            deferred_temporary_contexts.fetch_add(1);
        }
    }
    void RefreshTemporaryRouting() {
        bool needed = !temporary_views.empty();
        if (needed != counted) {
            if (needed) {
                deferred_temporary_contexts.fetch_add(1);
            } else {
                deferred_temporary_contexts.fetch_sub(1);
            }
            counted = needed;
        }
    }
    void QueryEnd(ClientContext &context, optional_ptr<ErrorData> error) override {
        if (error && error->HasError()) {
            RestoreMeasureViewSnapshots(pending);
            if (pending_temporary) {
                temporary_views = *pending_temporary;
            }
        } else if (context.transaction.HasActiveTransaction()) {
            for (auto &snapshot : pending) {
                transaction_snapshots.push_back(std::move(snapshot));
            }
            pending.clear();
        } else {
            FreeMeasureViewSnapshots(pending);
        }
        pending_temporary.reset();
        RefreshTemporaryRouting();
    }
    void TransactionCommit(MetaTransaction &, ClientContext &) override {
        FreeMeasureViewSnapshots(pending);
        FreeMeasureViewSnapshots(transaction_snapshots);
        pending_temporary.reset();
        transaction_temporary.reset();
        RefreshTemporaryRouting();
    }
    void TransactionRollback(MetaTransaction &, ClientContext &) override {
        RestoreMeasureViewSnapshots(pending);
        RestoreMeasureViewSnapshots(transaction_snapshots);
        if (transaction_temporary) {
            temporary_views = *transaction_temporary;
        }
        pending_temporary.reset();
        transaction_temporary.reset();
        RefreshTemporaryRouting();
    }
};

// Temporary entries belong to the originating session. A thread-local overlay
// exposes them to the lowerer without modifying another binder's catalog.
class DeferredTemporaryMeasureScope {
public:
    explicit DeferredTemporaryMeasureScope(DeferredMeasureCatalogState &state) {
        yardstick_push_measure_view_overlay();
        for (auto &entry : state.temporary_views) {
            yardstick_set_measure_view_overlay(entry.first.c_str(), entry.second.get());
        }
    }
    ~DeferredTemporaryMeasureScope() {
        yardstick_pop_measure_view_overlay();
    }
    void Save(const string &name) {
        yardstick_set_measure_view_overlay(name.c_str(), nullptr);
    }
    void Restore(const string &name) {
        yardstick_bypass_measure_view_overlay(name.c_str());
    }
};

static void SelectDeferredMeasureNamespace(const string &sql, DeferredMeasureCatalogState &state,
                                          DeferredTemporaryMeasureScope &scope) {
    for (auto &entry : state.temporary_views) {
        if (StatementReadsFromView(sql, entry.first, true)) {
            if (StatementTextReadsFromView(sql, entry.first, false)) {
                throw BinderException("A statement cannot combine temporary and permanent measure views named %s", entry.first);
            }
            scope.Restore(entry.first);
        }
    }
}

static BoundStatement BindDeferredMeasureStatement(ClientContext &context, Binder &parent,
                                                    const YardstickDeferredParseData &data) {
    NativeYardstickBindScope bind_scope(context);
    NativeYardstickParseScope parse_scope(nullptr, data.options);
    auto state = context.registered_state->GetOrCreate<DeferredMeasureCatalogState>("yardstick_deferred_catalog");
    DeferredTemporaryMeasureScope temporary_scope(*state);
    string sql = data.sql;
    string semantic_stripped;
    if (StartsWithSemantic(sql, semantic_stripped)) {
        sql = std::move(semantic_stripped);
    }
    try {
        SelectDeferredMeasureNamespace(sql, *state, temporary_scope);
        Parser original_parser(*CurrentNativeYardstickParserOptions());
        ParseNativeYardstickQuery(sql, original_parser);
        bool temporary_create = false;
        if (original_parser.statements.size() == 1 &&
            original_parser.statements[0]->type == StatementType::CREATE_STATEMENT) {
            auto &create = original_parser.statements[0]->Cast<CreateStatement>();
            if (create.info && create.info->type == CatalogType::VIEW_ENTRY) {
                auto &view = create.info->Cast<CreateViewInfo>();
                temporary_create = view.temporary;
                if (view.on_conflict == OnCreateConflict::IGNORE_ON_CONFLICT) {
                    // Let DuckDB resolve the exact target and skip an ignored
                    // body before Yardstick probes stars or changes metadata.
                    // Use the public binder API, including on Windows.
                    auto probe = original_parser.statements[0]->Copy();
                    auto binder = Binder::CreateBinder(context, &parent);
                    try {
                        auto bound = binder->Bind(*probe);
                        if (bound.plan->type == LogicalOperatorType::LOGICAL_CREATE_VIEW &&
                            bound.plan->Cast<LogicalCreate>().info->Cast<CreateViewInfo>().binding_mode ==
                                CreateViewBindingMode::SKIP_BINDING) {
                            return bound;
                        }
                    } catch (const Exception &) {
                        // An unignored body still contains measure markers.
                        // Its authoritative binding follows semantic lowering.
                    }
                }
            }
        }
        std::unique_ptr<YardstickCreateViewInfo, decltype(&yardstick_free_create_view_info)> native(
            FindNativeYardstickMeasures(sql.c_str()), yardstick_free_create_view_info);
        if (native && native->error) {
            throw BinderException(native->error);
        }
        if (native && native->is_measure_view) {
            if (temporary_create) {
                state->BeginTemporaryChange();
                temporary_scope.Save(native->view_name);
            } else {
                temporary_scope.Restore(native->view_name);
                state->pending.push_back(SnapshotMeasureView(native->view_name));
            }
            auto lowered = yardstick_process_create_view(sql.c_str());
            string error = lowered.error ? lowered.error : "";
            if (lowered.clean_sql) {
                sql = lowered.clean_sql;
            }
            yardstick_free_create_view_result(lowered);
            if (!error.empty()) {
                throw BinderException(error);
            }
            if (temporary_create) {
                state->temporary_views[StringUtil::Lower(native->view_name)] = std::shared_ptr<void>(
                    yardstick_snapshot_measure_view(native->view_name), yardstick_free_measure_view_snapshot);
            }
        } else {
            if (native) {
                // A plain replacement removes the old measure definition. A
                // temporary plain view also masks same-name permanent metadata.
                if (temporary_create) {
                    state->BeginTemporaryChange();
                    temporary_scope.Save(native->view_name);
                    state->temporary_views[StringUtil::Lower(native->view_name)] = std::shared_ptr<void>(
                        yardstick_snapshot_measure_view(native->view_name), yardstick_free_measure_view_snapshot);
                } else {
                    temporary_scope.Restore(native->view_name);
                    state->pending.push_back(SnapshotMeasureView(native->view_name));
                    yardstick_restore_measure_view_snapshot(native->view_name, yardstick_empty_measure_view_snapshot());
                }
            }
            DropViewInfo drop;
            if (ExtractDropViewInfoFromSql(sql, drop)) {
                auto temporary = state->temporary_views.find(StringUtil::Lower(drop.view_name));
                if (temporary != state->temporary_views.end() && drop.targets_temporary_view) {
                    state->BeginTemporaryChange();
                    state->temporary_views.erase(temporary);
                    temporary_scope.Restore(drop.view_name);
                } else {
                    temporary_scope.Restore(drop.view_name);
                    state->pending.push_back(SnapshotMeasureView(drop.view_name));
                    yardstick_drop_measure_view_from_sql(sql.c_str());
                }
            }
            if (yardstick_has_aggregate(sql.c_str())) {
                auto expanded = yardstick_expand_aggregate(sql.c_str());
                string error = expanded.error ? expanded.error : "";
                string warnings = AggregateWarnings(expanded);
                if (expanded.had_aggregate && expanded.expanded_sql) {
                    sql = expanded.expanded_sql;
                }
                yardstick_free_aggregate_result(expanded);
                if (!error.empty()) {
                    throw BinderException(error);
                }
                HandleAggregateWarnings(context, warnings);
            }
        }
        Parser parser(*CurrentNativeYardstickParserOptions());
        if (!ParseNativeYardstickQuery(sql, parser)) {
            parser.ParseQuery(sql);
        }
        if (parser.statements.size() != 1) {
            throw BinderException("Deferred Yardstick binding requires one statement");
        }
        auto binder = Binder::CreateBinder(context, &parent);
        return binder->Bind(*parser.statements[0]);
    } catch (...) {
        RestoreMeasureViewSnapshots(state->pending);
        if (state->pending_temporary) {
            state->temporary_views = *state->pending_temporary;
            state->pending_temporary.reset();
        }
        throw;
    }
}

static unique_ptr<TableRef> DeferredMeasureSelectBindReplace(ClientContext &context, TableFunctionBindInput &input) {
    auto &info = input.info->Cast<DeferredMeasureFunctionInfo>();
    NativeYardstickBindScope bind_scope(context);
    NativeYardstickParseScope parse_scope(info.parser_info.get(), context.GetParserOptions());
    auto state = context.registered_state->GetOrCreate<DeferredMeasureCatalogState>("yardstick_deferred_catalog");
    DeferredTemporaryMeasureScope temporary_scope(*state);
    string sql = input.inputs[0].GetValue<string>();
    SelectDeferredMeasureNamespace(sql, *state, temporary_scope);
    if (yardstick_has_aggregate(sql.c_str())) {
        auto expanded = yardstick_expand_aggregate(sql.c_str());
        string error = expanded.error ? expanded.error : "";
        string warnings = AggregateWarnings(expanded);
        if (expanded.had_aggregate && expanded.expanded_sql) {
            sql = expanded.expanded_sql;
        }
        yardstick_free_aggregate_result(expanded);
        if (!error.empty()) {
            throw BinderException(error);
        }
        HandleAggregateWarnings(context, warnings);
    }
    Parser parser(*CurrentNativeYardstickParserOptions());
    if (!ParseNativeYardstickQuery(sql, parser)) {
        parser.ParseQuery(sql);
    }
    if (parser.statements.size() != 1 || parser.statements[0]->type != StatementType::SELECT_STATEMENT) {
        throw BinderException("Deferred Yardstick query requires one SELECT statement");
    }
    return make_uniq<SubqueryRef>(unique_ptr_cast<SQLStatement, SelectStatement>(std::move(parser.statements[0])));
}
#endif

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
#if YARDSTICK_GRAMMAR_EXTENSION
                if (auto *deferred = dynamic_cast<YardstickDeferredParseData *>(state->parse_data.get())) {
                    return BindDeferredMeasureStatement(context, binder, *deferred);
                }
#endif
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

        // Non-yardstick extension statements should not be rewritten by yardstick.
        return {};
    }
    case StatementType::SELECT_STATEMENT: {
        auto sql_to_check = context.GetCurrentQuery();

        if (yardstick_has_aggregate(sql_to_check.c_str())) {
            YardstickAggregateResult result = yardstick_expand_aggregate(sql_to_check.c_str());
            if (result.error) {
                string error_msg(result.error);
                yardstick_free_aggregate_result(result);
                throw BinderException("Failed to expand AGGREGATE: %s", error_msg);
            }

            if (result.had_aggregate) {
                string expanded_sql(result.expanded_sql);
                string warnings = AggregateWarnings(result);
                yardstick_free_aggregate_result(result);

                // Rebind through table function so rewritten SQL executes with normal planning
                string wrapper_sql = YardstickWrapperSql(expanded_sql, warnings);
                Parser parser(YardstickParserOptions());
                parser.ParseQuery(wrapper_sql);
                auto statements = std::move(parser.statements);

                if (statements.empty()) {
                    throw BinderException("Table function wrapper produced no statements");
                }

                auto yardstick_binder = Binder::CreateBinder(context);
                return yardstick_binder->Bind(*statements[0]);
            }

            yardstick_free_aggregate_result(result);
        }
        return {};
    }
    default:
        return {};
    }
}

//=============================================================================
// EXTENSION LOADING
//=============================================================================

static void LoadInternal(ExtensionLoader &loader) {
    RegisterYardstickAggregateStateFunctions(loader);
    // Initialize parser FFI function pointers before registering query entry points.
    yardstick_init_parser_ffi(
        yardstick_find_aggregates,
        yardstick_free_aggregate_list,
        yardstick_parse_select,
        yardstick_free_select_info,
        yardstick_parse_expression,
        yardstick_free_expression_info,
        yardstick_parse_create_view,
        yardstick_free_create_view_info,
        yardstick_replace_range,
        yardstick_apply_replacements,
        yardstick_qualify_expression,
        yardstick_inline_order_by_subquery_aliases,
        yardstick_free_string,
        yardstick_expand_aggregate_call,
        yardstick_find_current_references,
        yardstick_free_current_reference_list,
        yardstick_current_where_is_single_valued,
        yardstick_expressions_equal,
        yardstick_find_query_scopes,
        yardstick_free_query_scopes,
        yardstick_rewrite_visible_filter,
        yardstick_decorate_measure,
        yardstick_window_marker,
        yardstick_rewrite_measure_windows
    );

    auto &db = loader.GetDatabaseInstance();
    auto &config = DBConfig::GetConfig(db);

    // Enable parser_override so yardstick intercepts queries before DuckDB's native parser.
    // FALLBACK mode: if our override doesn't handle the query, DuckDB's parser takes over.
    config.SetOptionByName("allow_parser_override_extension", Value("fallback"));

    // Register parser extension
    YardstickParserExtension parser;
#if YARDSTICK_GRAMMAR_EXTENSION
    parser.parser_info = RegisterYardstickGrammar(db);
#endif
    #if __has_include("duckdb/main/extension_callback_manager.hpp")
    ParserExtension::Register(config, parser);
    #else
    config.parser_extensions.push_back(parser);
    #endif

    // Register operator extension
    #if __has_include("duckdb/main/extension_callback_manager.hpp")
    OperatorExtension::Register(config, make_shared_ptr<YardstickOperatorExtension>());
    #else
    config.operator_extensions.push_back(make_uniq<YardstickOperatorExtension>());
    #endif

    // Register table function for AGGREGATE() expansion
    TableFunction query_func("yardstick", {LogicalType::VARCHAR},
                             YardstickQueryFunction, YardstickQueryBind);
#if YARDSTICK_GRAMMAR_EXTENSION
    query_func.function_info = make_shared_ptr<DeferredMeasureFunctionInfo>(parser.parser_info);
#endif
    loader.RegisterFunction(query_func);
    TableFunction query_func_with_warnings("yardstick", {LogicalType::VARCHAR, LogicalType::VARCHAR},
                                           YardstickQueryFunction, YardstickQueryBind);
#if YARDSTICK_GRAMMAR_EXTENSION
    query_func_with_warnings.function_info = make_shared_ptr<DeferredMeasureFunctionInfo>(parser.parser_info);
#endif
    loader.RegisterFunction(query_func_with_warnings);

#if YARDSTICK_GRAMMAR_EXTENSION
    TableFunction scoped_query("yardstick_scoped", {LogicalType::VARCHAR}, nullptr, nullptr);
    scoped_query.bind_replace = DeferredMeasureSelectBindReplace;
    scoped_query.function_info = make_shared_ptr<DeferredMeasureFunctionInfo>(parser.parser_info);
    loader.RegisterFunction(scoped_query);
#endif

    ScalarFunction warning_func("yardstick_warning", {LogicalType::VARCHAR}, LogicalType::BOOLEAN,
                                YardstickWarningFunction);
    warning_func.SetVolatile();
    warning_func.SetFallible();
    loader.RegisterFunction(warning_func);
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
