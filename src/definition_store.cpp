#include "definition_store.hpp"
#include "frontend_peg.hpp"
#include "yardstick_ffi.h"
#include "yardstick_parser_compat.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/schema_catalog_entry.hpp"
#include "duckdb/catalog/catalog_search_path.hpp"
#include "duckdb/catalog/entry_lookup_info.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/execution/executor.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/function/replacement_scan.hpp"
#include "duckdb/main/client_data.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/database_manager.hpp"
#include "duckdb/main/query_profiler.hpp"
#include "duckdb/main/client_context_state.hpp"
#include "duckdb/parser/parsed_data/create_table_info.hpp"
#include "duckdb/parser/parsed_data/create_view_info.hpp"
#include "duckdb/parser/parsed_data/drop_info.hpp"
#include "duckdb/parser/parsed_expression_iterator.hpp"
#include "duckdb/parser/expression/subquery_expression.hpp"
#include "duckdb/parser/expression/star_expression.hpp"
#include "duckdb/parser/query_node/select_node.hpp"
#include "duckdb/parser/statement/create_statement.hpp"
#include "duckdb/parser/statement/drop_statement.hpp"
#include "duckdb/parser/statement/explain_statement.hpp"
#include "duckdb/parser/statement/select_statement.hpp"
#include "duckdb/parser/tableref/basetableref.hpp"
#include "duckdb/parser/tableref/subqueryref.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/operator/logical_extension_operator.hpp"
#include "duckdb/planner/operator/logical_explain.hpp"
#include "duckdb/planner/planner_extension.hpp"
#if !YARDSTICK_GRAMMAR_EXTENSION
#include "duckdb/planner/operator/logical_create.hpp"
#include "duckdb/planner/operator/logical_simple.hpp"
#endif

#include <algorithm>
#include <cctype>
#include <set>
#include <type_traits>
#include <mutex>
#include <unordered_map>

extern "C" {
YardstickCreateViewResult yardstick_process_create_view(const char *sql);
void yardstick_free_create_view_result(YardstickCreateViewResult result);
YardstickAggregateResult yardstick_expand_aggregate(const char *sql);
void yardstick_free_aggregate_result(YardstickAggregateResult result);
void yardstick_free(char *ptr);
}

namespace duckdb {
namespace {
constexpr const char *DEFINITION_SETTING = "yardstick_definition_table";
std::mutex routing_lock;
std::unordered_map<const void *, idx_t> routing_sessions;
thread_local idx_t parsing_bypass = 0;
using StoredIdentifier = std::decay<decltype(std::declval<BaseTableRef>().alias)>::type;

struct ParsingBypass {
    ParsingBypass() { parsing_bypass++; }
    ~ParsingBypass() { parsing_bypass--; }
};

string QuoteIdentifier(const string &name) {
    return "\"" + StringUtil::Replace(name, "\"", "\"\"") + "\"";
}
string QuoteString(const string &value) {
    return "'" + StringUtil::Replace(value, "'", "''") + "'";
}

template <class T> string NameString(const T &name) {
#if YARDSTICK_NEW_EXPR_API
    return name.GetIdentifierName();
#else
    return name;
#endif
}

struct ViewName {
    string catalog;
    string schema;
    string name;

    string SQL() const {
        return QuoteIdentifier(catalog) + "." + QuoteIdentifier(schema) + "." + QuoteIdentifier(name);
    }
    bool Equals(const ViewName &other) const {
        return StringUtil::CIEquals(catalog, other.catalog) && StringUtil::CIEquals(schema, other.schema) &&
               StringUtil::CIEquals(name, other.name);
    }
};

ViewName GetName(const CreateInfo &info) {
#if YARDSTICK_NEW_EXPR_API
    auto &name = info.GetQualifiedName();
    return {NameString(name.Catalog()), NameString(name.Schema()), NameString(name.Name())};
#else
    auto &view = info.Cast<CreateViewInfo>();
    return {view.catalog, view.schema, view.view_name};
#endif
}

ViewName GetName(const BaseTableRef &ref) {
#if YARDSTICK_NEW_EXPR_API
    auto &name = ref.GetQualifiedName();
    return {NameString(name.Catalog()), NameString(name.Schema()), NameString(name.Name())};
#else
    return {ref.catalog_name, ref.schema_name, ref.table_name};
#endif
}

ViewName GetName(const DropInfo &info) {
#if YARDSTICK_NEW_EXPR_API
    auto &name = info.GetQualifiedName();
    return {NameString(name.Catalog()), NameString(name.Schema()), NameString(name.Name())};
#else
    return {info.catalog, info.schema, info.name};
#endif
}

QueryNode &SubqueryNode(SubqueryExpression &expression) {
#if YARDSTICK_NEW_EXPR_API
    return *expression.Subquery()->node;
#else
    return *expression.subquery->node;
#endif
}

optional_ptr<CatalogEntry> FindNativeEntry(ClientContext &context, const ViewName &name) {
#if YARDSTICK_NEW_EXPR_API
    EntryLookupInfo lookup(CatalogType::TABLE_ENTRY,
        QualifiedName(Identifier(name.catalog), Identifier(name.schema), Identifier(name.name)));
    return Catalog::GetEntry(context, lookup, OnEntryNotFound::RETURN_NULL);
#else
    EntryLookupInfo lookup(CatalogType::TABLE_ENTRY, name.name);
    return Catalog::GetEntry(context, name.catalog, name.schema, lookup, OnEntryNotFound::RETURN_NULL);
#endif
}

string SearchCatalog(const CatalogSearchEntry &entry) {
#if YARDSTICK_NEW_EXPR_API
    return NameString(entry.GetCatalog());
#else
    return entry.catalog;
#endif
}

string SearchSchema(const CatalogSearchEntry &entry) {
#if YARDSTICK_NEW_EXPR_API
    return NameString(entry.GetSchema());
#else
    return entry.schema;
#endif
}

vector<unique_ptr<SQLStatement>> Parse(const string &sql, const ParserOptions &options) {
    ParsingBypass bypass;
    auto internal_options = options;
#if YARDSTICK_GRAMMAR_EXTENSION
    if (auto current = CurrentNativeYardstickParserOptions()) internal_options = *current;
#endif
    internal_options.extensions = nullptr;
    Parser parser(internal_options);
    parser.ParseQuery(sql);
    return std::move(parser.statements);
}

string DefinitionTable(const ClientContext &context) {
    Value value;
    if (!context.TryGetCurrentSetting(DEFINITION_SETTING, value) || value.IsNull()) return "";
    return value.GetValue<string>();
}

ParserOptions DefinitionParserOptions(ClientContext &context) {
#if YARDSTICK_GRAMMAR_EXTENSION
    return YardstickContextParserOptions(context);
#else
    return context.GetParserOptions();
#endif
}

// This SELECT runs in the caller's transaction, including its uncommitted
// definition changes. A second Connection would read a different snapshot.
class DefinitionRowsSink : public PhysicalOperator {
public:
    DefinitionRowsSink(PhysicalPlan &plan, PhysicalOperator &scan, vector<vector<Value>> *rows)
        : PhysicalOperator(plan, PhysicalOperatorType::EXTENSION, {LogicalType::BOOLEAN}, 1), rows(rows) {
        children.push_back(scan);
    }
    vector<vector<Value>> *rows;
    bool IsSink() const override { return true; }
    bool IsSource() const override { return true; }
    SinkResultType Sink(ExecutionContext &, DataChunk &chunk, OperatorSinkInput &) const override {
        if (!rows) return SinkResultType::NEED_MORE_INPUT;
        for (idx_t row = 0; row < chunk.size(); row++) {
            vector<Value> values;
            for (idx_t col = 0; col < chunk.ColumnCount(); col++) values.push_back(chunk.GetValue(col, row));
            rows->push_back(std::move(values));
        }
        return SinkResultType::NEED_MORE_INPUT;
    }
    SourceResultType GetDataInternal(ExecutionContext &, DataChunk &, OperatorSourceInput &) const override {
        return SourceResultType::FINISHED;
    }
};

// Keep CREATE/REPLACE/DROP's empty success result instead of exposing the
// implementation's INSERT/UPDATE/DELETE row count to callers.
class LogicalDefinitionWrite : public LogicalExtensionOperator {
public:
    explicit LogicalDefinitionWrite(unique_ptr<LogicalOperator> write) {
        children.push_back(std::move(write));
    }
    string GetExtensionName() const override { return "yardstick"; }
    bool SupportSerialization() const override { return false; }
    vector<ColumnBinding> GetColumnBindings() override {
#if YARDSTICK_NEW_EXPR_API
        return {ColumnBinding(TableIndex(0), ProjectionIndex(0))};
#else
        return {ColumnBinding(0, 0)};
#endif
    }
    void ResolveTypes() override { types = {LogicalType::BOOLEAN}; }
    PhysicalOperator &CreatePlan(ClientContext &, PhysicalPlanGenerator &planner) override {
        auto &write = planner.CreatePlan(*children[0]);
        return planner.Make<DefinitionRowsSink>(write, nullptr);
    }
};

struct DefinitionProfilerScope {
    explicit DefinitionProfilerScope(ClientContext &context)
        : data(ClientData::Get(context)), previous(data.profiler) {
        data.profiler = make_shared_ptr<QueryProfiler>(context);
    }
    ~DefinitionProfilerScope() { data.profiler = std::move(previous); }
    ClientData &data;
    shared_ptr<QueryProfiler> previous;
};

vector<vector<Value>> ReadDefinitions(ClientContext &context, const string &table) {
    auto statements = Parse("SELECT view_catalog, view_schema, view_name, definition_sql FROM " + table,
                            YardstickBuiltinParserOptions());
    auto &select = statements[0]->Cast<SelectStatement>().node->Cast<SelectNode>();
    if (!FindNativeEntry(context, GetName(select.from_table->Cast<BaseTableRef>()))) return {};
    DefinitionProfilerScope profiler(context);
    auto binder = Binder::CreateBinder(context);
    auto bound = binder->Bind(*statements[0]);
    PhysicalPlanGenerator generator(context);
    auto plan = generator.Plan(std::move(bound.plan));
    vector<vector<Value>> rows;
    auto &sink = plan->Make<DefinitionRowsSink>(plan->Root(), &rows);
    {
        Executor executor(context);
        try {
            executor.Initialize(sink);
            while (!executor.ExecutionIsFinished()) {
                executor.ExecuteTask();
                if (!executor.ExecutionIsFinished()) executor.WaitForTask();
            }
            if (executor.HasError()) executor.ThrowException();
        } catch (...) {
            executor.CancelTasks();
            throw;
        }
        // Completed pipelines can still have worker tasks releasing their
        // state. Drain them before destroying this executor and its plan.
        executor.CancelTasks();
    }
    return rows;
}

struct Definition {
    ViewName name;
    string sql;
    string key;
    string clean_sql;
};

class DefinitionScope;
thread_local DefinitionScope *active_definitions = nullptr;

class DefinitionScope {
public:
    DefinitionScope(ClientContext &context, const string &table, const ParserOptions &options)
        : context(context), options(options), previous(active_definitions) {
        auto rows = ReadDefinitions(context, table);
        for (auto &row : rows) {
            for (auto &value : row) {
                if (value.IsNull() || value.type().id() != LogicalTypeId::VARCHAR) {
                    throw InvalidInputException("Yardstick definitions require non-NULL VARCHAR columns");
                }
            }
            Definition definition {{row[0].GetValue<string>(), row[1].GetValue<string>(), row[2].GetValue<string>()},
                                   row[3].GetValue<string>(), "__yardstick_definition_" + std::to_string(definitions.size()), ""};
            for (auto &existing : definitions) {
                if (definition.name.Equals(existing.name)) {
                    throw InvalidInputException("Duplicate Yardstick definition for %s in %s", definition.name.SQL(), table);
                }
            }
            definitions.push_back(std::move(definition));
        }
        yardstick_push_measure_view_namespace();
        active_definitions = this;
    }
    ~DefinitionScope() {
        active_definitions = previous;
        yardstick_pop_measure_view_namespace();
    }

    ClientContext &context;
    ParserOptions options;
    vector<Definition> definitions;
    DefinitionScope *previous;

    optional_ptr<Definition> Find(const ViewName &input) {
        for (auto &definition : definitions) {
            if (definition.key == input.name && input.catalog.empty() && input.schema.empty()) return &definition;
        }
        auto &path = *ClientData::Get(context).catalog_search_path;
        auto catalog = input.catalog;
        auto schema = input.schema;
        if (!schema.empty() && catalog.empty()) {
#if YARDSTICK_NEW_EXPR_API
            Identifier catalog_id, schema_id(schema);
            Binder::BindSchemaOrCatalog(context, catalog_id, schema_id);
            catalog = NameString(catalog_id);
            schema = NameString(schema_id);
#else
            Binder::BindSchemaOrCatalog(context, catalog, schema);
#endif
        }
        vector<ViewName> candidates;
        if (!catalog.empty() && !schema.empty()) {
            candidates.push_back({catalog, schema, input.name});
        } else if (!catalog.empty()) {
            candidates.push_back({catalog, NameString(path.GetDefaultSchema(StoredIdentifier(catalog))), input.name});
        } else if (!schema.empty()) {
            for (auto &entry : path.GetCatalogsForSchema(StoredIdentifier(schema))) candidates.push_back({NameString(entry), schema, input.name});
            if (candidates.empty()) candidates.push_back({NameString(DatabaseManager::GetDefaultDatabase(context)), schema, input.name});
        } else {
            auto &entry = path.GetDefault();
            candidates.push_back({SearchCatalog(entry), SearchSchema(entry), input.name});
            for (auto &entry : path.GetSetPaths()) {
                candidates.push_back({SearchCatalog(entry), SearchSchema(entry), input.name});
            }
        }
        for (auto &candidate : candidates) {
            if (candidate.catalog.empty()) candidate.catalog = NameString(DatabaseManager::GetDefaultDatabase(context));
            if (candidate.schema.empty()) candidate.schema = DEFAULT_SCHEMA;
            for (auto &definition : definitions) {
                if (definition.name.Equals(candidate)) return &definition;
            }
        }
        return nullptr;
    }

    void Hydrate() {
        for (auto &definition : definitions) {
            auto result = yardstick_process_create_view(definition.sql.c_str());
            string error = result.error ? result.error : "";
            if (result.clean_sql) definition.clean_sql = result.clean_sql;
            if (result.view_name) yardstick_rename_measure_view(result.view_name, definition.key.c_str());
            yardstick_free_create_view_result(result);
            if (!error.empty()) throw BinderException("Invalid stored Yardstick definition: %s", error);
        }
    }
};

// Some earlier parser extensions (notably the 1.5 CLI's autocomplete parser)
// accept ordinary SQL before Yardstick's parser override is reached. Track
// only routing and rebind requirements here; TLS-bearing namespaces must stay
// lexical so Prepare/ExtractPlan and cross-thread clients cannot retain them.
class DefinitionSessionState : public ClientContextState {
public:
    const void *parser_registry = nullptr;
    bool used_binding = false;

    ~DefinitionSessionState() override { SetRouting(nullptr); }

    void SetRouting(const void *registry) {
        if (parser_registry == registry) return;
        std::lock_guard<std::mutex> lock(routing_lock);
        if (parser_registry && --routing_sessions[parser_registry] == 0) routing_sessions.erase(parser_registry);
        parser_registry = registry;
        if (parser_registry) routing_sessions[parser_registry]++;
    }

    void QueryEnd() override { used_binding = false; }
};

unique_ptr<TableRef> ReplaceDefinition(ClientContext &context, ReplacementScanInput &input,
                                     optional_ptr<ReplacementScanData>) {
    if (DefinitionTable(context).empty()) return nullptr;
    unique_ptr<DefinitionScope> local_definitions;
    if (!active_definitions || &active_definitions->context != &context) {
        local_definitions = make_uniq<DefinitionScope>(context, DefinitionTable(context), DefinitionParserOptions(context));
        local_definitions->Hydrate();
    }
    auto definition = active_definitions->Find({input.catalog_name, input.schema_name, input.table_name});
    if (!definition) return nullptr;
    context.registered_state->GetOrCreate<DefinitionSessionState>("yardstick_definition_binding")->used_binding = true;
    auto statements = Parse(definition->clean_sql, YardstickBuiltinParserOptions());
    auto &create = statements[0]->Cast<CreateStatement>().info->Cast<CreateViewInfo>();
    auto result = make_uniq<SubqueryRef>(std::move(create.query));
    if (!create.aliases.empty()) {
        // Replacement scans overwrite their outer column aliases with the
        // caller's list. Keep the declaration's header on an inner relation.
        result->column_name_alias = create.aliases;
        result->alias = "__yardstick_view_body";
        auto select = make_uniq<SelectNode>();
        select->select_list.push_back(make_uniq<StarExpression>());
        select->from_table = std::move(result);
        auto query = make_uniq<SelectStatement>();
        query->node = std::move(select);
        result = make_uniq<SubqueryRef>(std::move(query));
    }
    result->alias = StoredIdentifier(definition->name.name);
    return std::move(result);
}

void SetDefinitionTable(ClientContext &context, SetScope scope, Value &value) {
    if (scope == SetScope::GLOBAL) throw InvalidInputException("yardstick_definition_table is a session setting");
    auto state = context.registered_state->GetOrCreate<DefinitionSessionState>("yardstick_definition_binding");
    if (value.IsNull() || value.GetValue<string>().empty()) {
        state->SetRouting(nullptr);
        return;
    }
    auto sql = "CREATE TABLE " + value.GetValue<string>() + " (view_catalog VARCHAR)";
    auto statements = Parse(sql, YardstickBuiltinParserOptions());
    if (statements.size() != 1 || statements[0]->type != StatementType::CREATE_STATEMENT ||
        statements[0]->Cast<CreateStatement>().info->type != CatalogType::TABLE_ENTRY) {
        throw InvalidInputException("yardstick_definition_table must be a catalog.schema.table name");
    }
    auto &info = statements[0]->Cast<CreateStatement>().info->Cast<CreateTableInfo>();
#if YARDSTICK_NEW_EXPR_API
    auto &name = info.GetQualifiedName();
    if (name.Path().size() != 3) throw InvalidInputException("yardstick_definition_table must be catalog.schema.table");
    value = Value(QuoteIdentifier(NameString(name.Catalog())) + "." + QuoteIdentifier(NameString(name.Schema())) + "." +
                  QuoteIdentifier(NameString(name.Name())));
#else
    if (info.catalog.empty() || info.schema.empty()) throw InvalidInputException("yardstick_definition_table must be catalog.schema.table");
    value = Value(QuoteIdentifier(info.catalog) + "." + QuoteIdentifier(info.schema) + "." + QuoteIdentifier(info.table));
#endif
    state->SetRouting(DefinitionParserOptions(context).extensions.get());
}

BoundStatement BindSQL(ClientContext &context, Binder &parent, const string &sql, const ParserOptions &options) {
    auto statements = Parse(sql, options);
    if (statements.size() != 1) throw BinderException("Expected one Yardstick statement");
    auto binder = Binder::CreateBinder(context, &parent);
    return binder->Bind(*statements[0]);
}

BoundStatement BindDefinitionWrite(ClientContext &context, Binder &parent, const string &sql) {
    auto bound = BindSQL(context, parent, sql, YardstickBuiltinParserOptions());
    bound.plan = make_uniq<LogicalDefinitionWrite>(std::move(bound.plan));
    bound.names = {"Success"};
    bound.types = {LogicalType::BOOLEAN};
    parent.GetStatementProperties().return_type = StatementReturnType::NOTHING;
    return bound;
}

string KeyPredicate(const ViewName &name) {
    return "lower(view_catalog) = lower(" + QuoteString(name.catalog) + ") AND lower(view_schema) = lower(" +
           QuoteString(name.schema) + ") AND lower(view_name) = lower(" + QuoteString(name.name) + ")";
}

string MaskMeasureDeclarations(const string &sql) {
    auto result = sql;
    auto tokens = Parser::Tokenize(sql);
    string previous;
    for (idx_t index = 0; index < tokens.size(); index++) {
        auto &token = tokens[index];
        if (token.type == SimplifiedTokenType::SIMPLIFIED_TOKEN_COMMENT) continue;
        auto end = token.start;
        while (end < sql.size() && (std::isalnum(static_cast<unsigned char>(sql[end])) || sql[end] == '_')) end++;
        auto word = StringUtil::Upper(sql.substr(token.start, end - token.start));
        auto next = index + 1;
        while (next < tokens.size() && tokens[next].type == SimplifiedTokenType::SIMPLIFIED_TOKEN_COMMENT) next++;
        if (previous == "AS" && word == "MEASURE" && next < tokens.size() &&
            tokens[next].type == SimplifiedTokenType::SIMPLIFIED_TOKEN_IDENTIFIER) {
            std::fill(result.begin() + token.start, result.begin() + end, ' ');
        }
        previous = std::move(word);
    }
    return result;
}

string FirstKeyword(const string &sql) {
    for (auto &token : Parser::Tokenize(sql)) {
        if (token.type == SimplifiedTokenType::SIMPLIFIED_TOKEN_COMMENT) continue;
        auto end = token.start;
        while (end < sql.size() && std::isalpha(static_cast<unsigned char>(sql[end]))) end++;
        return StringUtil::Upper(sql.substr(token.start, end - token.start));
    }
    return "";
}

idx_t ExplainBodyOffset(const string &sql) {
    idx_t depth = 0;
    for (auto &token : Parser::Tokenize(sql)) {
        if (token.type == SimplifiedTokenType::SIMPLIFIED_TOKEN_COMMENT ||
            token.type == SimplifiedTokenType::SIMPLIFIED_TOKEN_STRING_CONSTANT) continue;
        if (sql[token.start] == '(') { depth++; continue; }
        if (sql[token.start] == ')') { if (depth) depth--; continue; }
        if (depth) continue;
        auto end = token.start;
        while (end < sql.size() && std::isalpha(static_cast<unsigned char>(sql[end]))) end++;
        auto keyword = StringUtil::Upper(sql.substr(token.start, end - token.start));
        for (auto body : {"SELECT", "WITH", "VALUES", "CREATE", "DROP", "INSERT", "UPDATE", "DELETE", "COPY",
                          "EXECUTE", "CALL"}) {
            if (keyword == body) return token.start;
        }
    }
    throw ParserException("Expected a statement after EXPLAIN");
}

bool NeedsDefinitionLookup(ClientContext &context, const string &sql, const ParserOptions &options) {
    vector<unique_ptr<SQLStatement>> statements;
    try {
        statements = Parse(sql, options);
    } catch (const ParserException &) {
        return true; // Compatibility AT syntax is parsed after lowering.
    }
    if (statements.size() != 1 || statements[0]->type != StatementType::SELECT_STATEMENT) return true;
    bool needs_definitions = false;
    std::function<void(unique_ptr<ParsedExpression> &)> visit_expression;
    std::function<void(TableRef &)> visit_ref;
    visit_expression = [&](unique_ptr<ParsedExpression> &expression) {
        if (expression->GetExpressionClass() == ExpressionClass::SUBQUERY) {
            ParsedExpressionIterator::EnumerateQueryNodeChildren(
                SubqueryNode(expression->Cast<SubqueryExpression>()), visit_expression, visit_ref);
        }
        ParsedExpressionIterator::EnumerateChildren(*expression, visit_expression);
    };
    visit_ref = [&](TableRef &ref) {
        if (needs_definitions || ref.type != TableReferenceType::BASE_TABLE) return;
        auto entry = FindNativeEntry(context, GetName(ref.Cast<BaseTableRef>()));
        needs_definitions = !entry || entry->type == CatalogType::VIEW_ENTRY;
    };
    ParsedExpressionIterator::EnumerateQueryNodeChildren(*statements[0]->Cast<SelectStatement>().node,
                                                        visit_expression, visit_ref);
    return needs_definitions;
}

// Persist resolved source names, so USE/search_path changes in a later session
// cannot redirect a stored measure to an unrelated table. Keep declaration SQL
// intact, including AS MEASURE, comments, and header column aliases.
string QualifyDefinitionSources(ClientContext &context, DefinitionScope &definitions, const string &sql,
                                SelectStatement &query, bool temporary_store) {
    vector<std::pair<idx_t, std::pair<idx_t, string>>> replacements;
    std::set<string> cte_names;
    std::function<void(QueryNode &)> collect_ctes;
    std::function<void(unique_ptr<ParsedExpression> &)> collect_expression;
    collect_expression = [&](unique_ptr<ParsedExpression> &expression) {
        if (expression->GetExpressionClass() == ExpressionClass::SUBQUERY) {
            collect_ctes(SubqueryNode(expression->Cast<SubqueryExpression>()));
        }
        ParsedExpressionIterator::EnumerateChildren(*expression, collect_expression);
    };
    collect_ctes = [&](QueryNode &node) {
        for (auto &entry : node.cte_map.map) {
            cte_names.insert(StringUtil::Lower(NameString(entry.first)));
#if YARDSTICK_NEW_EXPR_API
            collect_ctes(*entry.second->query_node);
#else
            collect_ctes(*entry.second->query->node);
#endif
        }
        ParsedExpressionIterator::EnumerateQueryNodeChildren(node, collect_expression);
    };
    collect_ctes(*query.node);
    auto identifier_end = [&](idx_t start) {
        if (sql[start] == '"') {
            auto end = start + 1;
            while (end < sql.size()) {
                if (sql[end++] != '"') continue;
                if (end < sql.size() && sql[end] == '"') { end++; continue; }
                break;
            }
            return end;
        }
        auto end = start;
        while (end < sql.size()) {
            auto c = static_cast<unsigned char>(sql[end]);
            if (!std::isalnum(c) && c != '_' && c != '$' && c < 128) break;
            end++;
        }
        return end;
    };
    auto tokens = Parser::Tokenize(sql);
    std::function<void(unique_ptr<ParsedExpression> &)> visit_expression;
    std::function<void(TableRef &)> visit_ref;
    visit_expression = [&](unique_ptr<ParsedExpression> &expression) {
        if (expression->GetExpressionClass() == ExpressionClass::SUBQUERY) {
            ParsedExpressionIterator::EnumerateQueryNodeChildren(
                SubqueryNode(expression->Cast<SubqueryExpression>()), visit_expression, visit_ref);
        }
        ParsedExpressionIterator::EnumerateChildren(*expression, visit_expression);
    };
    visit_ref = [&](TableRef &ref) {
        if (ref.type != TableReferenceType::BASE_TABLE || !ref.query_location.IsValid()) return;
        auto input = GetName(ref.Cast<BaseTableRef>());
        if (input.catalog.empty() && input.schema.empty() && cte_names.count(StringUtil::Lower(input.name))) return;
        ViewName resolved;
        if (auto entry = FindNativeEntry(context, input)) {
            if (entry->temporary && !temporary_store) {
                throw BinderException("A persistent Yardstick definition cannot reference temporary table %s", input.name);
            }
            resolved = {NameString(entry->ParentCatalog().GetName()), NameString(entry->ParentSchema().name), NameString(entry->name)};
        } else if (auto definition = definitions.Find(input)) {
            resolved = definition->name;
        } else {
            return; // Leave replacement scans (including file paths) to DuckDB.
        }
#if YARDSTICK_NEW_EXPR_API
        idx_t start = ref.query_location.Start();
#else
        idx_t start = ref.query_location.GetIndex();
#endif
        auto end = identifier_end(start);
        // Tokenization supplies exact dot/name boundaries even when comments
        // or whitespace separate qualified-name components.
        idx_t components = 1 + !input.schema.empty() + !input.catalog.empty();
        auto next = std::find_if(tokens.begin(), tokens.end(), [&](const SimplifiedToken &token) { return token.start >= end; });
        while (components > 1 && next != tokens.end()) {
            if (next->type == SimplifiedTokenType::SIMPLIFIED_TOKEN_COMMENT) { next++; continue; }
            if (sql[next->start] == '.') { next++; continue; }
            end = identifier_end(next->start);
            components--;
            next++;
        }
        replacements.push_back({start, {end, resolved.SQL()}});
    };
    ParsedExpressionIterator::EnumerateQueryNodeChildren(*query.node, visit_expression, visit_ref);
    auto result = sql;
    std::sort(replacements.begin(), replacements.end(), [](const auto &a, const auto &b) { return a.first > b.first; });
    for (auto &replacement : replacements) {
        result.replace(replacement.first, replacement.second.first - replacement.first, replacement.second.second);
    }
    return result;
}
} // namespace

bool RouteYardstickDefinitionStatements(const string &query, const ParserOptions &options) {
    if (parsing_bypass) return false;
    if (StringUtil::Contains(StringUtil::Lower(query), DEFINITION_SETTING)) return true;
#if YARDSTICK_GRAMMAR_EXTENSION
    return options.context && !DefinitionTable(*options.context).empty();
#else
    std::lock_guard<std::mutex> lock(routing_lock);
    return routing_sessions.count(options.extensions.get()) != 0;
#endif
}
bool YardstickDefinitionParsingBypassed() { return parsing_bypass != 0; }
bool YardstickDefinitionStoreConfigured(const ClientContext &context) { return !DefinitionTable(context).empty(); }

unique_ptr<TableRef> YardstickDefinitionQueryRef(ClientContext &context, const string &sql,
                                                const ParserOptions &options, ParserExtensionInfo *parser_info) {
    if (DefinitionTable(context).empty()) return nullptr;
    ParsingBypass bypass;
#if YARDSTICK_GRAMMAR_EXTENSION
    NativeYardstickBindScope bind_scope(context);
    NativeYardstickParseScope parse_scope(parser_info, options);
#endif
    unique_ptr<DefinitionScope> local_definitions;
    if (!active_definitions || &active_definitions->context != &context) {
        local_definitions = make_uniq<DefinitionScope>(context, DefinitionTable(context), options);
        local_definitions->Hydrate();
    }
    auto temporary = EnterYardstickTemporaryDefinitions(context);
    auto expanded = yardstick_expand_aggregate(sql.c_str());
    string error = expanded.error ? expanded.error : "";
    string query = expanded.expanded_sql ? expanded.expanded_sql : sql;
    string warnings = expanded.warnings ? expanded.warnings : "";
    yardstick_free_aggregate_result(expanded);
    if (!error.empty()) throw BinderException(error);
    HandleAggregateWarnings(context, warnings);
    auto statements = Parse(query, options);
    if (statements.size() != 1 || statements[0]->type != StatementType::SELECT_STATEMENT) {
        throw BinderException("yardstick() requires one SELECT statement");
    }
    return make_uniq<SubqueryRef>(unique_ptr_cast<SQLStatement, SelectStatement>(std::move(statements[0])));
}

string YardstickStoredRelationName(const BaseTableRef &ref) {
    if (!active_definitions) return "";
    auto name = GetName(ref);
    auto definition = active_definitions->Find(name);
    if (!definition || FindNativeEntry(active_definitions->context, name)) return "";
    return definition ? definition->key : "";
}

void RegisterYardstickDefinitionStore(DBConfig &config) {
    config.AddExtensionOption(DEFINITION_SETTING,
        "Fully qualified table holding Yardstick view definitions; empty uses the legacy in-memory catalog",
        LogicalType::VARCHAR, Value(""), SetDefinitionTable);
    config.replacement_scans.emplace_back(ReplaceDefinition);
    PlannerExtension planner;
    planner.post_bind_function = [](PlannerExtensionInput &input, BoundStatement &statement) {
        auto state = input.context.registered_state->Get<DefinitionSessionState>("yardstick_definition_binding");
        if (state && state->used_binding) {
            input.binder.GetStatementProperties().always_require_rebind = true;
            state->used_binding = false;
        }
#if !YARDSTICK_GRAMMAR_EXTENSION
        if (parsing_bypass || DefinitionTable(input.context).empty()) return;
        input.binder.GetStatementProperties().always_require_rebind = true;
        string sql;
        ViewName name;
        auto target = statement.plan.get();
        optional_ptr<LogicalExplain> explanation;
        if (target->type == LogicalOperatorType::LOGICAL_EXPLAIN) {
            explanation = target->Cast<LogicalExplain>();
            target = explanation->children[0].get();
        }
        if (target->type == LogicalOperatorType::LOGICAL_CREATE_VIEW) {
            auto &view = target->Cast<LogicalCreate>().info->Cast<CreateViewInfo>();
            if (view.temporary) return;
            name = GetName(view);
            sql = view.ToString();
        } else if (target->type == LogicalOperatorType::LOGICAL_DROP) {
            auto &drop = target->Cast<LogicalSimple>().info->Cast<DropInfo>();
            if (drop.type != CatalogType::VIEW_ENTRY) return;
            name = GetName(drop);
            sql = drop.ToString();
        } else {
            return;
        }
        DefinitionScope definitions(input.context, DefinitionTable(input.context), input.context.GetParserOptions());
        if (!definitions.Find(name)) return;
        definitions.Hydrate();
        // The native binder planned a catalog write before discovering the
        // managed definition. Replace that write with the store's ordinary DML.
        input.binder.GetStatementProperties().modified_databases.clear();
        auto rewritten = BindYardstickDefinitionStatement(input.context, input.binder,
            YardstickDefinitionParseData(sql, input.context.GetParserOptions()), nullptr);
        if (explanation) {
            explanation->children[0] = std::move(rewritten.plan);
            explanation->logical_plan_unopt = explanation->children[0]->ToString(explanation->explain_format);
            input.binder.GetStatementProperties().return_type = StatementReturnType::QUERY_RESULT;
        } else {
            statement = std::move(rewritten);
        }
#endif
    };
    PlannerExtension::Register(config, std::move(planner));
}

BoundStatement BindYardstickDefinitionStatement(ClientContext &context, Binder &parent,
                                                const YardstickDefinitionParseData &data,
                                                ParserExtensionInfo *parser_info) {
    ParsingBypass bypass;
    auto table = DefinitionTable(context);
    if (table.empty()) {
        if (data.initialize) return BindSQL(context, parent, "SELECT true WHERE false", YardstickBuiltinParserOptions());
        return BindYardstickUnstoredStatement(context, parent, data.sql, data.options);
    }
#if YARDSTICK_GRAMMAR_EXTENSION
    NativeYardstickBindScope bind_scope(context);
    NativeYardstickParseScope parse_scope(parser_info, data.options);
#endif
    if (data.initialize) {
        auto create = StringUtil::StartsWith(StringUtil::Lower(table), "\"temp\".")
            ? "CREATE TEMP TABLE IF NOT EXISTS " : "CREATE TABLE IF NOT EXISTS ";
        return BindSQL(context, parent, create + table +
            " (view_catalog VARCHAR, view_schema VARCHAR, view_name VARCHAR, definition_sql VARCHAR)",
            YardstickBuiltinParserOptions());
    }
    if (FirstKeyword(data.sql) == "EXPLAIN") {
        auto body = ExplainBodyOffset(data.sql);
        auto parsed = Parse(data.sql.substr(0, body) + " SELECT true", YardstickBuiltinParserOptions());
        auto &explanation = parsed[0]->Cast<ExplainStatement>();
        auto child = BindYardstickDefinitionStatement(context, parent,
            YardstickDefinitionParseData(data.sql.substr(body), data.options), parser_info);
#if YARDSTICK_NEW_EXPR_API
        auto plan = make_uniq<LogicalExplain>(std::move(child.plan), explanation.explain_type, explanation.format);
        plan->logical_plan_unopt = plan->children[0]->ToString(context, explanation.format);
        if (explanation.explain_type == ExplainType::EXPLAIN_SQL) {
            plan->sql_output_names = std::move(child.names);
            plan->allow_unsupported_sql = explanation.allow_unsupported_sql;
        }
#else
        auto plan = make_uniq<LogicalExplain>(std::move(child.plan), explanation.explain_type, explanation.explain_format);
        plan->logical_plan_unopt = plan->children[0]->ToString(explanation.explain_format);
#endif
        child.plan = std::move(plan);
        child.names = {"explain_key", "explain_value"};
        child.types = {LogicalType::VARCHAR, LogicalType::VARCHAR};
        parent.GetStatementProperties().return_type = StatementReturnType::QUERY_RESULT;
        return child;
    }
    if (!NeedsDefinitionLookup(context, data.sql, data.options)) {
        return BindSQL(context, parent, data.sql, data.options);
    }
    parent.GetStatementProperties().always_require_rebind = true;
    unique_ptr<DefinitionScope> local_definitions;
    if (!active_definitions || &active_definitions->context != &context) {
        local_definitions = make_uniq<DefinitionScope>(context, table, data.options);
        local_definitions->Hydrate();
    }
    auto &definitions = *active_definitions;
    auto temporary_definitions = EnterYardstickTemporaryDefinitions(context);
    string error;
    // Context modifiers need semantic lowering before the builtin parser can
    // accept them. Only view DDL needs inspection before that lowering.
    auto keyword = FirstKeyword(data.sql);
    bool inspect_ddl = keyword == "CREATE" || keyword == "DROP";
    if (!inspect_ddl) {
        auto expanded = yardstick_expand_aggregate(data.sql.c_str());
        error = expanded.error ? expanded.error : "";
        string expanded_sql = expanded.expanded_sql ? expanded.expanded_sql : data.sql;
        string warnings = expanded.warnings ? expanded.warnings : "";
        yardstick_free_aggregate_result(expanded);
        if (!error.empty()) throw BinderException(error);
        HandleAggregateWarnings(context, warnings);
        return BindSQL(context, parent, expanded_sql, data.options);
    }
    auto statements = Parse(MaskMeasureDeclarations(data.sql), data.options);
    if (statements.size() != 1) throw BinderException("Expected one Yardstick statement");
    auto &statement = *statements[0];
    if (statement.type == StatementType::CREATE_STATEMENT &&
        statement.Cast<CreateStatement>().info->type == CatalogType::VIEW_ENTRY) {
        auto &view = statement.Cast<CreateStatement>().info->Cast<CreateViewInfo>();
        if (view.temporary) return BindYardstickUnstoredStatement(context, parent, data.sql, data.options);
        auto validator = Binder::CreateBinder(context);
        validator->SearchSchema(view);
        auto name = GetName(view);
        auto existing = definitions.Find(name);
        auto native = FindNativeEntry(context, name);
        if ((existing || native) && view.on_conflict == OnCreateConflict::IGNORE_ON_CONFLICT) {
            return BindDefinitionWrite(context, parent, "SELECT true WHERE false");
        }
        if (native) throw CatalogException("%s already exists outside the configured Yardstick definitions table", name.SQL());
        if (existing && view.on_conflict != OnCreateConflict::REPLACE_ON_CONFLICT) {
            throw CatalogException("Yardstick view %s already exists", name.SQL());
        }
        bool temporary_store = StringUtil::StartsWith(StringUtil::Lower(table), "\"temp\".");
        auto definition_sql = QualifyDefinitionSources(context, definitions, data.sql, *view.query, temporary_store);
        auto lowered = yardstick_process_create_view(definition_sql.c_str());
        error = lowered.error ? lowered.error : "";
        bool is_measure = lowered.is_measure_view;
        yardstick_free_create_view_result(lowered);
        if (!error.empty()) throw BinderException(error);
        if (!is_measure && !existing) return BindYardstickUnstoredStatement(context, parent, data.sql, data.options);
        // Validate the actual expressions, not the NULL placeholders emitted
        // for decomposable measures. Ordinary aggregates require the implicit
        // GROUP BY that AS MEASURE supplies.
        if (is_measure && view.query->node->type == QueryNodeType::SELECT_NODE) {
            auto &select = view.query->node->Cast<SelectNode>();
            bool has_window = false;
            std::function<void(unique_ptr<ParsedExpression> &)> inspect_expression;
            inspect_expression = [&](unique_ptr<ParsedExpression> &expression) {
                has_window |= expression->GetExpressionClass() == ExpressionClass::WINDOW;
                ParsedExpressionIterator::EnumerateChildren(*expression, inspect_expression);
            };
            for (auto &expression : select.select_list) inspect_expression(expression);
            if (select.groups.group_expressions.empty() && !has_window) {
                select.aggregate_handling = AggregateHandling::FORCE_AGGREGATES;
            }
        }
        validator->BindCreateViewInfo(view);
        auto sql = existing
            ? "UPDATE " + table + " SET definition_sql = " + QuoteString(definition_sql) + " WHERE " + KeyPredicate(name)
            : "INSERT INTO " + table + " (view_catalog, view_schema, view_name, definition_sql) VALUES (" +
              QuoteString(name.catalog) + ", " + QuoteString(name.schema) + ", " + QuoteString(name.name) + ", " + QuoteString(definition_sql) + ")";
        return BindDefinitionWrite(context, parent, sql);
    }
    if (statement.type == StatementType::DROP_STATEMENT) {
        auto &drop = *statement.Cast<DropStatement>().info;
        if (drop.type == CatalogType::VIEW_ENTRY) {
            auto definition = definitions.Find(GetName(drop));
            if (definition) {
                return BindDefinitionWrite(context, parent,
                    "DELETE FROM " + table + " WHERE " + KeyPredicate(definition->name));
            }
            return BindYardstickUnstoredStatement(context, parent, data.sql, data.options);
        }
    }
    auto expanded = yardstick_expand_aggregate(data.sql.c_str());
    error = expanded.error ? expanded.error : "";
    string expanded_sql = expanded.expanded_sql ? expanded.expanded_sql : data.sql;
    string warnings = expanded.warnings ? expanded.warnings : "";
    yardstick_free_aggregate_result(expanded);
    if (!error.empty()) throw BinderException(error);
    HandleAggregateWarnings(context, warnings);
    return BindSQL(context, parent, expanded_sql, data.options);
}
} // namespace duckdb
