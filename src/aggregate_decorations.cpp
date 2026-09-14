#include "aggregate_decorations.hpp"

#if YARDSTICK_GRAMMAR_EXTENSION
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/entry_lookup_info.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/parser/expression/columnref_expression.hpp"
#include "duckdb/parser/expression/conjunction_expression.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/parser/expression/lambda_expression.hpp"
#include "duckdb/parser/expression/star_expression.hpp"
#include "duckdb/parser/expression/subquery_expression.hpp"
#include "duckdb/parser/parsed_expression_iterator.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/parser/query_node/recursive_cte_node.hpp"
#include "duckdb/parser/query_node/select_node.hpp"
#include "duckdb/parser/query_node/set_operation_node.hpp"
#include "duckdb/parser/statement/select_statement.hpp"
#include "duckdb/parser/tableref/basetableref.hpp"
#include "duckdb/parser/tableref/expressionlistref.hpp"
#include "duckdb/parser/tableref/joinref.hpp"
#include "duckdb/parser/tableref/subqueryref.hpp"
#include "duckdb/parser/tableref/table_function_ref.hpp"
#include "duckdb/planner/binder.hpp"
#include <unordered_map>
#include <unordered_set>

namespace duckdb {
namespace {
using Names = std::unordered_set<string>;

unique_ptr<ParsedExpression> ParseOne(const string &sql, const ParserOptions &options) {
    auto expressions = Parser::ParseExpressionList(sql, options);
    if (expressions.size() != 1) {
        throw ParserException("Expected one expression in decorated AGGREGATE");
    }
    return std::move(expressions[0]);
}

struct DecorationScope {
    bool unqualified_is_local = true;
    bool inner_alias_shadowed = false;
    Names shadowed_qualifiers;
    Names shadowed_columns;
    Names aliases;
    Names lambda_parameters;
    vector<QueryNode *> cte_scopes;
};

// Call decorations are written against exposed dimensions. Resolve only names
// owned by the measure view; subquery and lambda bindings keep their own scope.
class DecorationReferences {
public:
    DecorationReferences(const vector<pair<string, string>> &dimensions, const vector<string> &qualifiers,
                         const ParserOptions &options, bool preserve_dimension_qualifiers)
        : preserve_dimension_qualifiers(preserve_dimension_qualifiers) {
        for (auto &qualifier : qualifiers) local_qualifiers.insert(StringUtil::Lower(qualifier));
        for (auto &dimension : dimensions) {
            auto expression = ParseOne(dimension.second, options);
            if (expression->HasSubquery()) {
                throw ParserException("Subquery dimensions are not supported in AGGREGATE decorations");
            }
            // Resolve base-row names without substituting exposed aliases again.
            QualifyDimension(expression, {}, preserve_dimension_qualifiers);
            dimension_expressions.emplace(StringUtil::Lower(dimension.first), std::move(expression));
        }
    }

    void Expression(unique_ptr<ParsedExpression> &expression, DecorationScope scope = {}) {
        if (!expression) return;
        if (expression->GetExpressionClass() == ExpressionClass::COLUMN_REF) {
            auto &names = expression->Cast<ColumnRefExpression>().ColumnNames();
            auto first = StringUtil::Lower(names[0].GetIdentifierName());
            if (scope.lambda_parameters.count(first)) return;
            bool local = names.size() == 1
                ? scope.unqualified_is_local && !scope.aliases.count(first) && !scope.shadowed_columns.count(first)
                : local_qualifiers.count(StringUtil::Lower(names[names.size() - 2].GetIdentifierName())) &&
                      !scope.shadowed_qualifiers.count(StringUtil::Lower(names[names.size() - 2].GetIdentifierName()));
            if (local) {
                if (scope.inner_alias_shadowed) {
                    throw ParserException("Conflicting recomputation alias in AGGREGATE decoration");
                }
                auto alias = expression->GetAlias();
                auto entry = dimension_expressions.find(StringUtil::Lower(names.back().GetIdentifierName()));
                if (entry == dimension_expressions.end() && preserve_dimension_qualifiers && names.size() == 1) {
                    // The window's consumer may provide this input through a
                    // join. Let its schema-aware lineage binding resolve it.
                    return;
                }
                expression = entry == dimension_expressions.end()
                    ? make_uniq<ColumnRefExpression>(names.back(), Identifier("_inner"))
                    : entry->second->Copy();
                expression->SetAlias(std::move(alias));
            }
            return;
        }
        if (expression->GetExpressionClass() == ExpressionClass::LAMBDA) {
            auto &lambda = expression->Cast<LambdaExpression>();
            if (AddLambdaBindings(lambda, scope.lambda_parameters)) {
                Expression(lambda.RightMutable(), scope);
                return;
            }
        }
        if (expression->GetExpressionClass() == ExpressionClass::SUBQUERY) {
            auto &subquery = expression->Cast<SubqueryExpression>();
            Expression(subquery.GetChildMutable(), scope);
            Query(*subquery.SubqueryMutable()->node, scope);
            return;
        }
        ParsedExpressionIterator::EnumerateChildren(*expression,
            [&](unique_ptr<ParsedExpression> &child) { Expression(child, scope); });
    }

private:
    bool preserve_dimension_qualifiers;
    Names local_qualifiers;
    std::unordered_map<string, unique_ptr<ParsedExpression>> dimension_expressions;

    static bool AddLambdaBindings(LambdaExpression &lambda, Names &bindings) {
        // Arrow syntax also means JSON access; only the lambda keyword has
        // unambiguous lexical bindings before DuckDB's binder runs.
        if (lambda.GetLambdaSyntaxType() != LambdaSyntaxType::LAMBDA_KEYWORD) return false;
        string error;
        auto parameters = lambda.ExtractColumnRefExpressions(error);
        if (!error.empty()) throw ParserException(error);
        for (auto &parameter : parameters) {
            auto &names = parameter.get().Cast<ColumnRefExpression>().ColumnNames();
            if (names.size() != 1) throw ParserException("Invalid AGGREGATE decoration lambda parameter");
            bindings.insert(StringUtil::Lower(names[0].GetIdentifierName()));
        }
        return true;
    }

    static void QualifyDimension(unique_ptr<ParsedExpression> &expression, Names bindings,
                                 bool preserve_dimension_qualifiers) {
        if (expression->GetExpressionClass() == ExpressionClass::COLUMN_REF) {
            auto &names = expression->Cast<ColumnRefExpression>().ColumnNamesMutable();
            if (!bindings.count(StringUtil::Lower(names[0].GetIdentifierName()))) {
                if (preserve_dimension_qualifiers) {
                    // Window recomputation projects against the original FROM,
                    // where joined relations may expose the same column name.
                    names.insert(names.begin(), Identifier("_inner"));
                } else {
                    auto column = names.back();
                    names = {Identifier("_inner"), std::move(column)};
                }
            }
            return;
        }
        if (expression->GetExpressionClass() == ExpressionClass::LAMBDA) {
            auto &lambda = expression->Cast<LambdaExpression>();
            if (AddLambdaBindings(lambda, bindings)) {
                QualifyDimension(lambda.RightMutable(), bindings, preserve_dimension_qualifiers);
                return;
            }
        }
        ParsedExpressionIterator::EnumerateChildren(*expression,
            [&](unique_ptr<ParsedExpression> &child) {
                QualifyDimension(child, bindings, preserve_dimension_qualifiers);
            });
    }

    static void Qualifiers(const TableRef &table, Names &names) {
        if (!table.alias.empty()) {
            names.insert(StringUtil::Lower(table.alias.GetIdentifierName()));
        } else if (table.type == TableReferenceType::BASE_TABLE) {
            names.insert(StringUtil::Lower(table.Cast<BaseTableRef>().Table().GetIdentifierName()));
        } else if (table.type == TableReferenceType::TABLE_FUNCTION) {
            auto &function = table.Cast<TableFunctionRef>().function;
            if (function->GetExpressionClass() == ExpressionClass::FUNCTION) {
                names.insert(StringUtil::Lower(function->Cast<FunctionExpression>().FunctionName().GetIdentifierName()));
            }
        }
        if (table.type == TableReferenceType::JOIN) {
            auto &join = table.Cast<JoinRef>();
            Qualifiers(*join.left, names);
            Qualifiers(*join.right, names);
        }
    }

    static bool ProjectionNames(QueryNode &query, vector<Identifier> &names) {
        if (query.type == QueryNodeType::SET_OPERATION_NODE) {
            return ProjectionNames(*query.Cast<SetOperationNode>().children[0], names);
        }
        if (query.type != QueryNodeType::SELECT_NODE) return false;
        auto &select = query.Cast<SelectNode>();
        for (auto &expression : select.select_list) {
            bool expands = false;
            ParsedExpressionIterator::VisitExpressionClass(*expression, ExpressionClass::STAR,
                [&](const ParsedExpression &) { expands = true; });
            ParsedExpressionIterator::VisitExpression<FunctionExpression>(*expression,
                [&](const FunctionExpression &function) {
                    expands |= StringUtil::CIEquals(function.FunctionName().GetIdentifierName(), "unnest");
                });
            if (expands) return false;
            names.push_back(expression->GetName());
        }
        return true;
    }

    static Names NamedColumns(vector<Identifier> names, const vector<Identifier> &aliases) {
        for (idx_t i = 0; i < aliases.size() && i < names.size(); ++i) names[i] = aliases[i];
        Names result;
        for (auto &name : names) result.insert(StringUtil::Lower(name.GetIdentifierName()));
        return result;
    }

    static Names InputColumns(TableRef &table, const DecorationScope &scope) {
        if (table.type == TableReferenceType::JOIN) {
            // ON conditions can correlate to the outer row. They do not change
            // the set of input column names; USING only removes duplicates.
            auto &join = table.Cast<JoinRef>();
            auto names = InputColumns(*join.left, scope);
            auto right = InputColumns(*join.right, scope);
            names.insert(right.begin(), right.end());
            return names;
        }
        if (table.type == TableReferenceType::SUBQUERY) {
            vector<Identifier> names;
            if (ProjectionNames(*table.Cast<SubqueryRef>().subquery->node, names)) {
                return NamedColumns(std::move(names), table.column_name_alias);
            }
        }
        if (table.type == TableReferenceType::BASE_TABLE) {
            auto &name = table.Cast<BaseTableRef>().GetQualifiedName();
            if (name.Path().size() == 1) {
                for (auto it = scope.cte_scopes.rbegin(); it != scope.cte_scopes.rend(); ++it) {
                    auto entry = (*it)->cte_map.map.find(name.Name());
                    if (entry == (*it)->cte_map.map.end()) continue;
                    vector<Identifier> names;
                    if (ProjectionNames(*entry->second->query_node, names)) {
                        for (idx_t i = 0; i < entry->second->aliases.size() && i < names.size(); ++i) {
                            names[i] = entry->second->aliases[i];
                        }
                        return NamedColumns(std::move(names), table.column_name_alias);
                    }
                    break;
                }
            }
        }
        auto context = CurrentNativeYardstickClientContext();
        if (!context) {
            throw BinderException("AGGREGATE decoration column resolution requires the originating bind context");
        }
        // Binding a layout-only probe discovers table/function/CTE output names
        // without evaluating the filter. A FROM clause shadows only its actual
        // columns, not every unqualified reference in a correlated expression.
        auto probe = make_uniq<SelectNode>();
        probe->select_list.push_back(make_uniq<StarExpression>());
        probe->from_table = table.Copy();
        for (auto it = scope.cte_scopes.rbegin(); it != scope.cte_scopes.rend(); ++it) {
            for (auto &entry : (*it)->cte_map.map) {
                if (probe->cte_map.map.find(entry.first) == probe->cte_map.map.end()) {
                    probe->cte_map.map.insert(entry.first, entry.second->Copy());
                }
            }
        }
        auto binder = Binder::CreateBinder(*context);
        auto bound = binder->Bind(*probe);
        Names names;
        for (auto &name : bound.names) names.insert(StringUtil::Lower(name.GetIdentifierName()));
        return names;
    }

    static DecorationScope WithBindings(TableRef &table, DecorationScope scope) {
        if (table.type == TableReferenceType::EMPTY_FROM) return scope;
        auto columns = InputColumns(table, scope);
        scope.shadowed_columns.insert(columns.begin(), columns.end());
        Qualifiers(table, scope.shadowed_qualifiers);
        scope.inner_alias_shadowed |= scope.shadowed_qualifiers.count("_inner") != 0;
        return scope;
    }

    void Query(QueryNode &query, DecorationScope outer) {
        outer.cte_scopes.push_back(&query);
        for (auto &entry : query.cte_map.map) {
            if (!entry.second->query_node) throw ParserException("Unsupported CTE in AGGREGATE decoration");
            Query(*entry.second->query_node, outer);
        }
        auto scope = outer;
        switch (query.type) {
        case QueryNodeType::SELECT_NODE: {
            auto &select = query.Cast<SelectNode>();
            if (select.from_table) {
                scope = WithBindings(*select.from_table, outer);
                Table(*select.from_table, outer);
            }
            for (auto &item : select.select_list) {
                Expression(item, scope);
                if (item->HasAlias()) scope.aliases.insert(StringUtil::Lower(item->GetAlias().GetIdentifierName()));
            }
            for (auto &group : select.groups.group_expressions) Expression(group, scope);
            Expression(select.where_clause, scope);
            Expression(select.having, scope);
            Expression(select.qualify, scope);
            break;
        }
        case QueryNodeType::SET_OPERATION_NODE:
            for (auto &child : query.Cast<SetOperationNode>().children) Query(*child, outer);
            scope.unqualified_is_local = false;
            break;
        case QueryNodeType::RECURSIVE_CTE_NODE: {
            auto &cte = query.Cast<RecursiveCTENode>();
            Query(*cte.left, outer);
            Query(*cte.right, outer);
            scope.unqualified_is_local = false;
            for (auto &key : cte.key_targets) Expression(key, scope);
            break;
        }
        default:
            throw ParserException("Unsupported query in AGGREGATE decoration");
        }
        ParsedExpressionIterator::EnumerateQueryNodeModifiers(query,
            [&](unique_ptr<ParsedExpression> &expression) { Expression(expression, scope); });
    }

    void Table(TableRef &table, DecorationScope scope) {
        switch (table.type) {
        case TableReferenceType::JOIN: {
            auto &join = table.Cast<JoinRef>();
            Table(*join.left, scope);
            Table(*join.right, WithBindings(*join.left, scope));
            Expression(join.condition, WithBindings(table, scope));
            break;
        }
        case TableReferenceType::SUBQUERY:
            Query(*table.Cast<SubqueryRef>().subquery->node, scope);
            break;
        case TableReferenceType::EXPRESSION_LIST:
            for (auto &row : table.Cast<ExpressionListRef>().values)
                for (auto &expression : row) Expression(expression, scope);
            break;
        case TableReferenceType::TABLE_FUNCTION: {
            auto &function = table.Cast<TableFunctionRef>();
            Expression(function.function, scope);
            if (function.subquery) Query(*function.subquery->node, scope);
            break;
        }
        case TableReferenceType::BASE_TABLE:
        case TableReferenceType::EMPTY_FROM:
            break;
        default:
            throw ParserException("Unsupported table in AGGREGATE decoration");
        }
    }
};

bool IsAggregate(const FunctionExpression &function) {
    if (auto context = CurrentNativeYardstickClientContext()) {
        EntryLookupInfo lookup(CatalogType::AGGREGATE_FUNCTION_ENTRY, function.GetQualifiedName());
        auto entry = Catalog::GetEntry(*context, lookup, OnEntryNotFound::RETURN_NULL);
        return entry && entry->type == CatalogType::AGGREGATE_FUNCTION_ENTRY;
    }
    return IsYardstickStandardAggregate(function.FunctionName().GetIdentifierName());
}

class AggregateDecorator {
public:
    explicit AggregateDecorator(FunctionExpression &call) : call(call) {
    }

    void Expression(unique_ptr<ParsedExpression> &expression) {
        if (expression->GetExpressionClass() == ExpressionClass::SUBQUERY ||
            expression->GetExpressionClass() == ExpressionClass::WINDOW) return;
        if (expression->GetExpressionClass() == ExpressionClass::FUNCTION) {
            auto &function = expression->Cast<FunctionExpression>();
            if (IsAggregate(function)) {
                function.DistinctMutable() |= call.Distinct();
                if (call.Filter()) {
                    function.FilterMutable() = function.Filter()
                        ? make_uniq<ConjunctionExpression>(ExpressionType::CONJUNCTION_AND,
                              std::move(function.FilterMutable()), call.Filter()->Copy())
                        : call.Filter()->Copy();
                }
                if (call.OrderBy() && !call.OrderBy()->orders.empty()) {
                    auto ordering = make_uniq<OrderModifier>();
                    for (auto &order : call.OrderBy()->orders) {
                        ordering->orders.emplace_back(order.type, order.null_order, order.expression->Copy());
                    }
                    // The caller chooses primary ordering; declaration ordering
                    // remains deterministic for ties in those call-level keys.
                    if (function.OrderBy()) {
                        for (auto &order : function.OrderBy()->orders) {
                            ordering->orders.emplace_back(order.type, order.null_order, order.expression->Copy());
                        }
                    }
                    function.OrderByMutable() = std::move(ordering);
                }
                aggregate_count++;
                if (call.ExportState()) {
                    if (function.ExportState()) {
                        throw ParserException("Cannot export an already exported measure aggregate");
                    }
                    function.ExportStateMutable() = true;
                    auto field = "s" + std::to_string(states.size());
                    states.emplace_back(Identifier(field), expression->Copy());
                    expression = make_uniq<ColumnRefExpression>(Identifier(field));
                }
                return;
            }
        }
        ParsedExpressionIterator::EnumerateChildren(*expression,
            [&](unique_ptr<ParsedExpression> &child) { Expression(child); });
    }

    idx_t aggregate_count = 0;
    vector<FunctionArgument> states;

private:
    FunctionExpression &call;
};
} // namespace

string DecorateYardstickMeasureExpression(const string &measure_expression, const string &call_sql,
                                         const vector<pair<string, string>> &dimensions,
                                         const vector<string> &local_qualifiers, const ParserOptions &options,
                                         bool preserve_dimension_qualifiers) {
    auto expression = ParseOne(measure_expression, options);
    auto parsed_call = ParseOne(call_sql, options);
    if (parsed_call->GetExpressionClass() != ExpressionClass::FUNCTION) {
        throw ParserException("Expected an AGGREGATE call for measure decorations");
    }
    auto &call = parsed_call->Cast<FunctionExpression>();
    if (!StringUtil::CIEquals(call.FunctionName().GetIdentifierName(), "aggregate") ||
        call.GetArguments().size() != 1) {
        throw ParserException("Expected a single-argument AGGREGATE call for measure decorations");
    }
    DecorationReferences references(dimensions, local_qualifiers, options, preserve_dimension_qualifiers);
    references.Expression(call.FilterMutable());
    if (call.OrderBy()) {
        for (auto &order : call.OrderByMutable()->orders) references.Expression(order.expression);
    }
    bool direct_aggregate = expression->GetExpressionClass() == ExpressionClass::FUNCTION &&
        IsAggregate(expression->Cast<FunctionExpression>());
    AggregateDecorator decorator(call);
    decorator.Expression(expression);
    if (!decorator.aggregate_count) {
        throw ParserException("Decorated AGGREGATE requires a measure containing aggregate functions");
    }
    if (call.ExportState()) {
        if (direct_aggregate) return decorator.states[0].GetExpression().ToString();
        vector<unique_ptr<ParsedExpression>> arguments;
        arguments.push_back(ConstantExpression::String(expression->ToString()));
        arguments.push_back(make_uniq<FunctionExpression>(Identifier("struct_pack"), std::move(decorator.states)));
        expression = make_uniq<FunctionExpression>(Identifier("yardstick_state"), std::move(arguments));
    }
    return expression->ToString();
}
} // namespace duckdb
#endif
