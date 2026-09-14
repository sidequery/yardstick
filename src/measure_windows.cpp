#include "measure_windows.hpp"

#if YARDSTICK_GRAMMAR_EXTENSION
#include "frontend_peg.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/entry_lookup_info.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/parser/expression/columnref_expression.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/parser/expression/lambda_expression.hpp"
#include "duckdb/parser/expression/star_expression.hpp"
#include "duckdb/parser/expression/subquery_expression.hpp"
#include "duckdb/parser/expression/window_expression.hpp"
#include "duckdb/parser/parsed_data/create_view_info.hpp"
#include "duckdb/parser/parsed_expression_iterator.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/parser/query_node/select_node.hpp"
#include "duckdb/parser/query_node/set_operation_node.hpp"
#include "duckdb/parser/query_node/recursive_cte_node.hpp"
#include "duckdb/parser/statement/create_statement.hpp"
#include "duckdb/parser/statement/select_statement.hpp"
#include "duckdb/parser/tableref/basetableref.hpp"
#include "duckdb/parser/tableref/joinref.hpp"
#include "duckdb/parser/tableref/subqueryref.hpp"
#include "duckdb/parser/tableref/expressionlistref.hpp"
#include "duckdb/parser/tableref/table_function_ref.hpp"
#include "duckdb/planner/binder.hpp"

#include <functional>
#include <unordered_map>
#include <unordered_set>

namespace duckdb {
namespace {

using ExpressionMap = std::unordered_map<string, unique_ptr<ParsedExpression>>;
using Names = std::unordered_set<string>;

string Key(const string &text) {
    return StringUtil::Lower(text);
}

string Quote(const string &text) {
    return "\"" + StringUtil::Replace(text, "\"", "\"\"") + "\"";
}

unique_ptr<ParsedExpression> Expression(const string &sql, const ParserOptions &options) {
    auto expressions = Parser::ParseExpressionList(sql, options);
    if (expressions.size() != 1) {
        throw ParserException("Expected one expression while rewriting a measure window");
    }
    return std::move(expressions[0]);
}

unique_ptr<ParsedExpression> ModifierExpression(const string &sql, const ParserOptions &options) {
    auto references = FindNativeYardstickCurrentReferences(sql.c_str());
    if (!references) {
        return Expression(sql, options);
    }
    string normalized = sql;
    string error = references->error ? references->error : "";
    for (idx_t i = references->count; i > 0; i--) {
        auto &reference = references->references[i - 1];
        normalized.replace(reference.start_pos, reference.end_pos - reference.start_pos,
                           "current(" + string(reference.dimension) + ")");
    }
    yardstick_free_current_reference_list(references);
    if (!error.empty()) {
        throw ParserException(error);
    }
    return Expression(normalized, options);
}

unique_ptr<QueryNode> Query(const string &sql, const ParserOptions &options,
                           vector<Identifier> *view_aliases = nullptr) {
    Parser parser(options);
    parser.ParseQuery(sql);
    if (parser.statements.size() != 1) {
        throw ParserException("Expected one query while rewriting a measure window");
    }
    auto &statement = *parser.statements[0];
    if (statement.type == StatementType::CREATE_STATEMENT) {
        auto &info = statement.Cast<CreateStatement>().info->Cast<CreateViewInfo>();
        if (view_aliases) {
            *view_aliases = info.aliases;
        }
        return info.query->node->Copy();
    }
    return statement.Cast<SelectStatement>().node->Copy();
}

void Walk(unique_ptr<ParsedExpression> &expression,
          const std::function<bool(unique_ptr<ParsedExpression> &)> &visit, Names lambda_parameters = {}) {
    if (expression && expression->GetExpressionClass() == ExpressionClass::COLUMN_REF &&
        lambda_parameters.count(Key(expression->Cast<ColumnRefExpression>().ColumnNames()[0].GetIdentifierName()))) {
        return;
    }
    if (!expression || !visit(expression) || expression->GetExpressionClass() == ExpressionClass::SUBQUERY) {
        return;
    }
    if (expression->GetExpressionClass() == ExpressionClass::LAMBDA) {
        auto &lambda = expression->Cast<LambdaExpression>();
        if (lambda.GetLambdaSyntaxType() == LambdaSyntaxType::LAMBDA_KEYWORD) {
            string error;
            auto parameters = lambda.ExtractColumnRefExpressions(error);
            if (!error.empty()) {
                throw ParserException(error);
            }
            for (auto &parameter : parameters) {
                lambda_parameters.insert(Key(parameter.get().Cast<ColumnRefExpression>()
                                                 .GetColumnName().GetIdentifierName()));
            }
            Walk(lambda.RightMutable(), visit, std::move(lambda_parameters));
            return;
        }
    }
    ParsedExpressionIterator::EnumerateChildren(*expression, [&](unique_ptr<ParsedExpression> &child) {
        Walk(child, visit, lambda_parameters);
    });
}

void SelectExpressions(SelectNode &select,
                       const std::function<void(unique_ptr<ParsedExpression> &)> &visit) {
    for (auto &expression : select.select_list) {
        visit(expression);
    }
    for (auto &expression : select.groups.group_expressions) {
        visit(expression);
    }
    if (select.having) {
        visit(select.having);
    }
    if (select.qualify) {
        visit(select.qualify);
    }
    ParsedExpressionIterator::EnumerateQueryNodeModifiers(select, visit);
}

ExpressionMap Aliases(const SelectNode &select) {
    ExpressionMap aliases;
    for (auto &expression : select.select_list) {
        if (expression->HasAlias()) {
            auto copy = expression->Copy();
            copy->ClearAlias();
            aliases.emplace(Key(expression->GetAlias().GetIdentifierName()), std::move(copy));
        }
    }
    return aliases;
}

void ExpandAliases(unique_ptr<ParsedExpression> &expression, const ExpressionMap &aliases,
                   const Names &input_columns = {}, Names expanding = {}) {
    if (!expression || expression->GetExpressionClass() == ExpressionClass::SUBQUERY) {
        return;
    }
    if (expression->GetExpressionClass() == ExpressionClass::LAMBDA) {
        auto &lambda = expression->Cast<LambdaExpression>();
        if (lambda.GetLambdaSyntaxType() == LambdaSyntaxType::LAMBDA_KEYWORD) {
            auto bindings = input_columns;
            string error;
            auto parameters = lambda.ExtractColumnRefExpressions(error);
            if (!error.empty()) {
                throw ParserException(error);
            }
            for (auto &parameter : parameters) {
                bindings.insert(Key(parameter.get().Cast<ColumnRefExpression>().GetColumnName().GetIdentifierName()));
            }
            ExpandAliases(lambda.RightMutable(), aliases, bindings, expanding);
            return;
        }
    }
    if (expression->GetExpressionClass() == ExpressionClass::COLUMN_REF) {
        auto &column = expression->Cast<ColumnRefExpression>();
        if (!column.IsQualified()) {
            auto key = Key(column.GetColumnName().GetIdentifierName());
            auto match = aliases.find(key);
            if (match != aliases.end() && !input_columns.count(key) && !expanding.count(key)) {
                expanding.insert(key);
                auto alias = expression->GetAlias();
                expression = match->second->Copy();
                expression->SetAlias(alias);
                ExpandAliases(expression, aliases, input_columns, std::move(expanding));
                return;
            }
        }
    }
    ParsedExpressionIterator::EnumerateChildren(*expression, [&](unique_ptr<ParsedExpression> &child) {
        ExpandAliases(child, aliases, input_columns, expanding);
    });
}

bool HasAggregate(unique_ptr<ParsedExpression> &expression) {
    bool found = false;
    Walk(expression, [&](unique_ptr<ParsedExpression> &node) {
        if (node->GetExpressionClass() == ExpressionClass::WINDOW) {
            return false;
        }
        if (node->GetExpressionClass() == ExpressionClass::FUNCTION) {
            auto &function = node->Cast<FunctionExpression>();
            if (auto context = CurrentNativeYardstickClientContext()) {
                EntryLookupInfo lookup(CatalogType::AGGREGATE_FUNCTION_ENTRY, function.GetQualifiedName());
                auto entry = Catalog::GetEntry(*context, lookup, OnEntryNotFound::RETURN_NULL);
                found |= entry && entry->type == CatalogType::AGGREGATE_FUNCTION_ENTRY;
            } else {
                found |= IsYardstickStandardAggregate(function.FunctionName().GetIdentifierName());
            }
        }
        return true;
    });
    return found;
}

Names InputColumns(SelectNode &select) {
    Names result;
    if (!select.from_table || select.from_table->type == TableReferenceType::EMPTY_FROM) {
        return result;
    }
    auto context = CurrentNativeYardstickClientContext();
    if (!context) {
        throw BinderException("Measure window binding requires the originating bind context");
    }
    auto probe = select.Copy();
    auto &layout = probe->Cast<SelectNode>();
    layout.select_list.clear();
    layout.select_list.push_back(make_uniq<StarExpression>());
    layout.groups = GroupByNode();
    layout.having.reset();
    layout.qualify.reset();
    layout.where_clause.reset();
    layout.modifiers.clear();
    layout.aggregate_handling = AggregateHandling::STANDARD_HANDLING;
    auto binder = Binder::CreateBinder(*context);
    auto bound = binder->Bind(*probe);
    for (auto &name : bound.names) {
        result.insert(Key(name.GetIdentifierName()));
    }
    return result;
}

void BaseQualifiers(const TableRef &table, Names &names) {
    if (table.type == TableReferenceType::BASE_TABLE && table.alias.empty()) {
        names.insert(Key(table.Cast<BaseTableRef>().Table().GetIdentifierName()));
    } else if (table.type == TableReferenceType::JOIN) {
        auto &join = table.Cast<JoinRef>();
        BaseQualifiers(*join.left, names);
        BaseQualifiers(*join.right, names);
    }
}

bool IsGrouped(SelectNode &select) {
    if (!select.groups.group_expressions.empty() || !select.groups.grouping_sets.empty() ||
        select.aggregate_handling == AggregateHandling::FORCE_AGGREGATES) {
        return true;
    }
    for (auto &expression : select.select_list) {
        if (HasAggregate(expression)) {
            return true;
        }
    }
    return select.having && HasAggregate(select.having);
}

void ExpandStars(SelectNode &select) {
    auto contains_star = [](unique_ptr<ParsedExpression> &expression) {
        bool found = false;
        Walk(expression, [&](unique_ptr<ParsedExpression> &node) {
            if (node->GetExpressionClass() == ExpressionClass::WINDOW ||
                (node->GetExpressionClass() == ExpressionClass::FUNCTION &&
                 IsYardstickStandardAggregate(node->Cast<FunctionExpression>().FunctionName().GetIdentifierName()))) {
                return false;
            }
            found |= node->GetExpressionClass() == ExpressionClass::STAR;
            return true;
        });
        return found;
    };
    bool has_star = false;
    for (auto &projection : select.select_list) {
        has_star |= contains_star(projection);
    }
    if (!has_star) {
        return;
    }
    auto context = CurrentNativeYardstickClientContext();
    if (!context) {
        throw BinderException("Measure window star expansion requires the originating bind context");
    }
    auto probe = select.Copy();
    auto &layout = probe->Cast<SelectNode>();
    layout.groups = GroupByNode();
    layout.having.reset();
    layout.qualify.reset();
    layout.where_clause.reset();
    layout.modifiers.clear();
    layout.aggregate_handling = AggregateHandling::STANDARD_HANDLING;
    string prefix = "__ys_star_projection_";
    auto source_sql = select.ToString();
    while (source_sql.find(prefix) != string::npos) {
        prefix += "_";
    }
    std::unordered_map<string, idx_t> placeholders;
    for (idx_t i = 0; i < layout.select_list.size(); i++) {
        if (!contains_star(layout.select_list[i])) {
            auto name = prefix + std::to_string(i);
            auto placeholder = ConstantExpression::FromValue(Value());
            placeholder->SetAlias(Identifier(name));
            layout.select_list[i] = std::move(placeholder);
            placeholders.emplace(name, i);
        }
    }
    auto binder = Binder::CreateBinder(*context);
    auto bound = binder->Bind(*probe);
    Names base_qualifiers;
    if (select.from_table) {
        BaseQualifiers(*select.from_table, base_qualifiers);
    }
    vector<unique_ptr<ParsedExpression>> expanded;
    for (auto &projection : bound.extra_info.original_expressions) {
        auto found = placeholders.find(projection->GetAlias().GetIdentifierName());
        if (projection->GetExpressionClass() == ExpressionClass::CONSTANT && found != placeholders.end()) {
            expanded.push_back(std::move(select.select_list[found->second]));
        } else {
            Walk(projection, [&](unique_ptr<ParsedExpression> &node) {
                if (node->GetExpressionClass() != ExpressionClass::COLUMN_REF) {
                    return true;
                }
                auto &names = node->Cast<ColumnRefExpression>().ColumnNamesMutable();
                if (names.size() < 2) {
                    return false;
                }
                auto qualifier = names[names.size() - 2].GetIdentifierName();
                string visible_name;
                auto qualifier_key = Key(qualifier);
                for (auto &candidate : base_qualifiers) {
                    if ((qualifier_key == candidate || StringUtil::EndsWith(qualifier_key, "." + candidate)) &&
                        candidate.size() > visible_name.size()) {
                        visible_name = candidate;
                    }
                }
                if (!visible_name.empty()) {
                    auto column = names.back();
                    names = {Identifier(visible_name), std::move(column)};
                }
                return false;
            });
            expanded.push_back(std::move(projection));
        }
    }
    if (expanded.size() != bound.names.size()) {
        throw BinderException("Unable to expand measure window projection columns");
    }
    select.select_list = std::move(expanded);
}

unique_ptr<TableRef> Relation(const string &name) {
    auto relation = make_uniq<BaseTableRef>();
    relation->SetTable(Identifier(name));
    return std::move(relation);
}

unique_ptr<TableRef> Subquery(unique_ptr<QueryNode> query, const string &alias) {
    auto statement = make_uniq<SelectStatement>();
    statement->node = std::move(query);
    return make_uniq<SubqueryRef>(std::move(statement), Identifier(alias));
}

struct SourcePlan {
    const MeasureWindowSource *spec;
    string base_name;
    string lineage_name;
    string id_name;
    unique_ptr<QueryNode> base;
    unique_ptr<QueryNode> view;
    vector<unique_ptr<ParsedExpression>> columns;
    std::unordered_map<string, idx_t> column_index;
    ExpressionMap dimensions;
    Names qualifiers;
    Names input_columns;
};

struct ReferenceScope {
    bool unqualified_is_outer = true;
    Names local_qualifiers;
    Names local_columns;
    Names aliases;
    Names lambda_parameters;
    vector<QueryNode *> cte_scopes;
};

void RelationQualifiers(const TableRef &table, Names &names) {
    if (!table.alias.empty()) {
        names.insert(Key(table.alias.GetIdentifierName()));
    } else if (table.type == TableReferenceType::BASE_TABLE) {
        names.insert(Key(table.Cast<BaseTableRef>().Table().GetIdentifierName()));
    } else if (table.type == TableReferenceType::TABLE_FUNCTION) {
        auto &function = table.Cast<TableFunctionRef>().function;
        if (function->GetExpressionClass() == ExpressionClass::FUNCTION) {
            names.insert(Key(function->Cast<FunctionExpression>().FunctionName().GetIdentifierName()));
        }
    }
    if (table.type == TableReferenceType::JOIN) {
        auto &join = table.Cast<JoinRef>();
        RelationQualifiers(*join.left, names);
        RelationQualifiers(*join.right, names);
    }
}

// Only references captured from the surrounding base-row scope are changed.
// Lambda parameters and nested query relations retain their own namespaces.
class CapturedReferences {
public:
    using Visitor = std::function<void(unique_ptr<ParsedExpression> &)>;

    CapturedReferences(const Names &qualifiers_p, Visitor visitor_p)
        : qualifiers(qualifiers_p), visitor(std::move(visitor_p)) {
    }

    void Expression(unique_ptr<ParsedExpression> &expression, ReferenceScope scope = {}) {
        if (!expression) {
            return;
        }
        if (expression->GetExpressionClass() == ExpressionClass::COLUMN_REF) {
            auto &names = expression->Cast<ColumnRefExpression>().ColumnNames();
            auto first = Key(names[0].GetIdentifierName());
            if (scope.lambda_parameters.count(first)) {
                return;
            }
            bool captured = names.size() == 1
                ? scope.unqualified_is_outer && !scope.aliases.count(first) && !scope.local_columns.count(first)
                : (scope.unqualified_is_outer || qualifiers.count(first)) && !scope.local_qualifiers.count(first);
            if (captured) {
                visitor(expression);
            }
            return;
        }
        if (expression->GetExpressionClass() == ExpressionClass::FUNCTION &&
            StringUtil::CIEquals(expression->Cast<FunctionExpression>().FunctionName().GetIdentifierName(), "current")) {
            return;
        }
        if (expression->GetExpressionClass() == ExpressionClass::LAMBDA) {
            auto &lambda = expression->Cast<LambdaExpression>();
            if (lambda.GetLambdaSyntaxType() == LambdaSyntaxType::LAMBDA_KEYWORD) {
                string error;
                auto parameters = lambda.ExtractColumnRefExpressions(error);
                if (!error.empty()) {
                    throw ParserException(error);
                }
                for (auto &parameter : parameters) {
                    scope.lambda_parameters.insert(Key(parameter.get().Cast<ColumnRefExpression>()
                                                           .GetColumnName().GetIdentifierName()));
                }
                Expression(lambda.RightMutable(), std::move(scope));
                return;
            }
        }
        if (expression->GetExpressionClass() == ExpressionClass::SUBQUERY) {
            auto &subquery = expression->Cast<SubqueryExpression>();
            Expression(subquery.GetChildMutable(), scope);
            Query(*subquery.SubqueryMutable()->node, scope);
            return;
        }
        ParsedExpressionIterator::EnumerateChildren(*expression, [&](unique_ptr<ParsedExpression> &child) {
            Expression(child, scope);
        });
    }

private:
    ReferenceScope WithBindings(TableRef &table, ReferenceScope scope) {
        if (table.type != TableReferenceType::EMPTY_FROM) {
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
            auto context = CurrentNativeYardstickClientContext();
            if (!context) {
                throw BinderException("Measure window correlation requires the originating bind context");
            }
            auto binder = Binder::CreateBinder(*context);
            auto bound = binder->Bind(*probe);
            for (auto &name : bound.names) {
                scope.local_columns.insert(Key(name.GetIdentifierName()));
            }
            RelationQualifiers(table, scope.local_qualifiers);
        }
        return scope;
    }

    void Query(QueryNode &query, ReferenceScope scope) {
        scope.cte_scopes.push_back(&query);
        for (auto &entry : query.cte_map.map) {
            if (entry.second->query_node) {
                Query(*entry.second->query_node, scope);
            }
        }
        if (query.type == QueryNodeType::SELECT_NODE) {
            auto &select = query.Cast<SelectNode>();
            if (select.from_table) {
                Table(*select.from_table, scope);
                scope = WithBindings(*select.from_table, std::move(scope));
            }
            for (auto &projection : select.select_list) {
                Expression(projection, scope);
                if (projection->HasAlias()) {
                    scope.aliases.insert(Key(projection->GetAlias().GetIdentifierName()));
                }
            }
            for (auto &group : select.groups.group_expressions) {
                Expression(group, scope);
            }
            Expression(select.where_clause, scope);
            Expression(select.having, scope);
            Expression(select.qualify, scope);
        } else if (query.type == QueryNodeType::SET_OPERATION_NODE) {
            for (auto &child : query.Cast<SetOperationNode>().children) {
                Query(*child, scope);
            }
            scope.unqualified_is_outer = false;
        } else if (query.type == QueryNodeType::RECURSIVE_CTE_NODE) {
            auto &cte = query.Cast<RecursiveCTENode>();
            Query(*cte.left, scope);
            Query(*cte.right, scope);
            scope.unqualified_is_outer = false;
        }
        ParsedExpressionIterator::EnumerateQueryNodeModifiers(query, [&](unique_ptr<ParsedExpression> &expression) {
            Expression(expression, scope);
        });
    }

    void Table(TableRef &table, ReferenceScope scope) {
        if (table.type == TableReferenceType::JOIN) {
            auto &join = table.Cast<JoinRef>();
            Table(*join.left, scope);
            Table(*join.right, WithBindings(*join.left, scope));
            Expression(join.condition, WithBindings(table, scope));
        } else if (table.type == TableReferenceType::SUBQUERY) {
            Query(*table.Cast<SubqueryRef>().subquery->node, scope);
        } else if (table.type == TableReferenceType::EXPRESSION_LIST) {
            for (auto &row : table.Cast<ExpressionListRef>().values) {
                for (auto &expression : row) {
                    Expression(expression, scope);
                }
            }
        } else if (table.type == TableReferenceType::TABLE_FUNCTION) {
            auto &function = table.Cast<TableFunctionRef>();
            Expression(function.function, scope);
            if (function.subquery) {
                Query(*function.subquery->node, scope);
            }
        }
    }

    const Names &qualifiers;
    Visitor visitor;
};

unique_ptr<ParsedExpression> BaseColumn(const ParsedExpression &expression, const SourcePlan &plan) {
    auto result = expression.Copy();
    result->ClearAlias();
    auto &names = result->Cast<ColumnRefExpression>().ColumnNamesMutable();
    if (names.size() > 1 && StringUtil::CIEquals(names[0].GetIdentifierName(), "_inner")) {
        names.erase(names.begin());
    } else if (names.size() == 2 &&
               (StringUtil::CIEquals(names[0].GetIdentifierName(), plan.spec->relation_name) ||
                StringUtil::CIEquals(names[0].GetIdentifierName(), plan.spec->alias))) {
        names.erase(names.begin());
    }
    if (names.size() == 1) {
        auto key = Key(names[0].GetIdentifierName());
        auto dimension = plan.dimensions.find(key);
        if (!plan.input_columns.count(key) && dimension != plan.dimensions.end()) {
            return dimension->second->Copy();
        }
    }
    return result;
}

void NormalizeModifierReferences(unique_ptr<ParsedExpression> &expression, const SourcePlan &plan) {
    CapturedReferences references(plan.qualifiers, [&](unique_ptr<ParsedExpression> &node) {
        auto &column = node->Cast<ColumnRefExpression>();
        auto match = plan.dimensions.find(Key(column.GetColumnName().GetIdentifierName()));
        auto replacement = match == plan.dimensions.end() ? BaseColumn(*node, plan) : match->second->Copy();
        CapturedReferences base_references(plan.qualifiers, [&](unique_ptr<ParsedExpression> &base) {
            auto &names = base->Cast<ColumnRefExpression>().ColumnNamesMutable();
            names.insert(names.begin(), Identifier("_inner"));
        });
        base_references.Expression(replacement);
        node = std::move(replacement);
    });
    references.Expression(expression);
}

bool IsSourceReference(const ParsedExpression &expression, const SourcePlan &plan, bool call_scope = false) {
    auto &names = expression.Cast<ColumnRefExpression>().ColumnNames();
    if (names.size() > 1) {
        auto first = Key(names[0].GetIdentifierName());
        if (call_scope) {
            // Decorated source keys carry _inner. A defining FROM alias can be
            // reused by a different relation in the consumer's namespace.
            return first == "_inner" || first == Key(plan.spec->relation_name) ||
                   first == Key(plan.spec->alias);
        }
        return plan.qualifiers.count(first) || plan.input_columns.count(first) || plan.dimensions.count(first);
    }
    auto name = Key(names[0].GetIdentifierName());
    return plan.input_columns.count(name) || plan.dimensions.count(name);
}

void CollectColumns(unique_ptr<ParsedExpression> &expression, SourcePlan &plan, bool call_scope = false) {
    CapturedReferences references(plan.qualifiers, [&](unique_ptr<ParsedExpression> &node) {
        if (!IsSourceReference(*node, plan, call_scope)) {
            return;
        }
        auto base = BaseColumn(*node, plan);
        auto key = Key(base->ToString());
        if (!plan.column_index.count(key)) {
            plan.column_index.emplace(key, plan.columns.size());
            plan.columns.push_back(std::move(base));
        }
    });
    references.Expression(expression);
}

void RebindColumns(unique_ptr<ParsedExpression> &expression, const SourcePlan &plan,
                   const string &qualifier = "") {
    CapturedReferences references(plan.qualifiers, [&](unique_ptr<ParsedExpression> &node) {
        auto base = BaseColumn(*node, plan);
        auto match = plan.column_index.find(Key(base->ToString()));
        if (match != plan.column_index.end()) {
            auto alias = node->GetAlias();
            auto name = Identifier("__ys_c" + std::to_string(match->second));
            node = qualifier.empty() ? make_uniq<ColumnRefExpression>(name)
                                     : make_uniq<ColumnRefExpression>(name, Identifier(qualifier));
            node->SetAlias(alias);
        }
    });
    references.Expression(expression);
}

SourcePlan BuildSource(const MeasureWindowSource &source, const vector<MeasureWindowCall> &calls,
                       const std::unordered_map<string, idx_t> &caller_order_counts,
                       idx_t index, Names &cte_names, const ParserOptions &options) {
    SourcePlan plan;
    plan.spec = &source;
    auto prefix = "__ys_window_source_" + std::to_string(index);
    plan.lineage_name = prefix + "_lineage";
    plan.id_name = prefix + "_id";
    vector<Identifier> view_aliases;
    plan.view = Query(source.clean_select_sql, options, &view_aliases);
    for (auto &entry : plan.view->cte_map.map) {
        cte_names.insert(Key(entry.first.GetIdentifierName()));
    }
    plan.base_name = prefix + "_base";
    idx_t suffix = 0;
    while (!cte_names.insert(Key(plan.base_name)).second) {
        plan.base_name = prefix + "_base_" + std::to_string(++suffix);
    }
    auto &view = plan.view->Cast<SelectNode>();
    if (view.from_table) {
        RelationQualifiers(*view.from_table, plan.qualifiers);
    }
    plan.qualifiers.insert("_inner");
    plan.qualifiers.insert(Key(source.relation_name));
    plan.qualifiers.insert(Key(source.alias));
    ExpandStars(view);
    auto input_columns = InputColumns(view);
    plan.input_columns = input_columns;
    auto aliases = Aliases(view);
    for (auto &dimension : source.dimensions) {
        plan.dimensions.emplace(Key(dimension.first), Expression(dimension.second, options));
    }
    // Resolve output aliases before rebinding the defining query to generated
    // base columns. Keep names stable even for originally unaliased dimensions.
    for (auto &expression : view.select_list) {
        auto name = expression->GetName();
        ExpandAliases(expression, aliases, input_columns);
        expression->SetAlias(name);
    }
    for (auto &expression : view.groups.group_expressions) {
        ExpandAliases(expression, aliases, input_columns);
    }
    if (view.having) {
        ExpandAliases(view.having, aliases, input_columns);
    }
    if (view.qualify) {
        ExpandAliases(view.qualify, aliases, input_columns);
    }
    for (auto &modifier : view.modifiers) {
        if (modifier->type == ResultModifierType::ORDER_MODIFIER) {
            for (auto &order : modifier->Cast<OrderModifier>().orders) {
                auto &expression = order.expression;
                bool output_alias = expression->GetExpressionClass() == ExpressionClass::COLUMN_REF &&
                    !expression->Cast<ColumnRefExpression>().IsQualified() &&
                    aliases.count(Key(expression->Cast<ColumnRefExpression>().GetColumnName().GetIdentifierName()));
                // A bare ORDER BY name chooses an output alias; names inside
                // an ORDER BY expression retain normal input precedence.
                ExpandAliases(expression, aliases, output_alias ? Names {} : input_columns);
            }
        } else if (modifier->type == ResultModifierType::DISTINCT_MODIFIER) {
            for (auto &expression : modifier->Cast<DistinctModifier>().distinct_on_targets) {
                bool output_alias = expression->GetExpressionClass() == ExpressionClass::COLUMN_REF &&
                    !expression->Cast<ColumnRefExpression>().IsQualified() &&
                    aliases.count(Key(expression->Cast<ColumnRefExpression>().GetColumnName().GetIdentifierName()));
                ExpandAliases(expression, aliases, output_alias ? Names {} : input_columns);
            }
        }
    }
    // Legacy metadata stores only explicitly aliased dimensions. Recover the
    // complete dimension set from the defining projection for AT contexts.
    for (idx_t i = 0; i < view.select_list.size(); i++) {
        auto &projection = view.select_list[i];
        if (HasAggregate(projection)) {
            continue;
        }
        bool window = false;
        Walk(projection, [&](unique_ptr<ParsedExpression> &node) {
            window |= node->GetExpressionClass() == ExpressionClass::WINDOW;
            return !window;
        });
        if (window) {
            continue;
        }
        bool references_base = false;
        CapturedReferences references(plan.qualifiers, [&](unique_ptr<ParsedExpression> &) { references_base = true; });
        references.Expression(projection);
        if (references_base) {
            auto name = i < view_aliases.size() ? view_aliases[i] : projection->GetName();
            auto expression = projection->Copy();
            expression->ClearAlias();
            plan.dimensions[Key(name.GetIdentifierName())] = std::move(expression);
        }
    }
    SelectExpressions(view, [&](unique_ptr<ParsedExpression> &expression) { CollectColumns(expression, plan); });
    for (auto &dimension : plan.dimensions) {
        CollectColumns(dimension.second, plan);
    }
    for (auto &call : calls) {
        if (call.source_key != source.key) {
            continue;
        }
        auto expression = Expression(call.expression_sql, options);
        auto order_count = caller_order_counts.at(call.marker_name);
        if (order_count) {
            Walk(expression, [&](unique_ptr<ParsedExpression> &node) {
                if (node->GetExpressionClass() != ExpressionClass::FUNCTION) {
                    return true;
                }
                auto &function = node->Cast<FunctionExpression>();
                if (function.OrderBy() && function.OrderBy()->orders.size() >= order_count) {
                    auto &orders = function.OrderByMutable()->orders;
                    for (idx_t i = 0; i < order_count; i++) {
                        CollectColumns(orders[i].expression, plan, true);
                    }
                    // Only the caller's prefix uses the consumer namespace.
                    // Declaration arguments, filters and order ties retain the
                    // defining FROM namespace, even when aliases are reused.
                    orders.erase(orders.begin(), orders.begin() + order_count);
                }
                return true;
            });
        }
        CollectColumns(expression, plan);
        for (auto &modifier : call.modifiers) {
            if (modifier.type == WindowContextType::WHERE) {
                auto condition = ModifierExpression(modifier.value, options);
                NormalizeModifierReferences(condition, plan);
                CollectColumns(condition, plan);
            }
        }
    }
    auto base = make_uniq<SelectNode>();
    base->from_table = std::move(view.from_table);
    base->where_clause = std::move(view.where_clause);
    base->cte_map = std::move(view.cte_map);
    base->sample = std::move(view.sample);
    auto identity = Expression("row_number() OVER ()", options);
    identity->SetAlias(Identifier(plan.id_name));
    base->select_list.push_back(std::move(identity));
    for (idx_t i = 0; i < plan.columns.size(); i++) {
        auto column = plan.columns[i]->Copy();
        column->SetAlias(Identifier("__ys_c" + std::to_string(i)));
        base->select_list.push_back(std::move(column));
    }
    plan.base = std::move(base);
    bool grouped = source.grouped || IsGrouped(view);
    // Adding provenance to SELECT DISTINCT would otherwise prevent duplicate
    // elimination. Equal visible rows form one lineage group instead.
    if (!grouped) {
        for (auto it = view.modifiers.begin(); it != view.modifiers.end();) {
            if ((*it)->type == ResultModifierType::DISTINCT_MODIFIER &&
                (*it)->Cast<DistinctModifier>().distinct_on_targets.empty()) {
                GroupingSet set;
                for (idx_t i = 0; i < view.select_list.size(); i++) {
                    auto group = view.select_list[i]->Copy();
                    group->ClearAlias();
                    view.groups.group_expressions.push_back(std::move(group));
                    set.insert(ProjectionIndex(i));
                }
                view.groups.grouping_sets.push_back(std::move(set));
                it = view.modifiers.erase(it);
                grouped = true;
            } else {
                ++it;
            }
        }
    }
    SelectExpressions(view, [&](unique_ptr<ParsedExpression> &expression) { RebindColumns(expression, plan); });
    for (idx_t i = 0; i < view_aliases.size() && i < view.select_list.size(); i++) {
        view.select_list[i]->SetAlias(view_aliases[i]);
    }
    view.from_table = Relation(plan.base_name);
    auto lineage = Expression((grouped ? "list(" : "list_value(") + Quote(plan.id_name) + ")", options);
    lineage->SetAlias(Identifier(plan.lineage_name));
    view.select_list.push_back(std::move(lineage));
    return plan;
}

bool ReplaceSource(unique_ptr<TableRef> &relation, const SourcePlan &plan) {
    if (!relation) {
        return false;
    }
    if (relation->type == TableReferenceType::JOIN) {
        auto &join = relation->Cast<JoinRef>();
        bool left = ReplaceSource(join.left, plan);
        bool right = ReplaceSource(join.right, plan);
        return left || right;
    }
    if (relation->type != TableReferenceType::BASE_TABLE) {
        return false;
    }
    auto &table = relation->Cast<BaseTableRef>();
    auto alias = relation->alias.empty() ? table.Table().GetIdentifierName() : relation->alias.GetIdentifierName();
    auto expected_alias = plan.spec->alias.empty() ? plan.spec->relation_name : plan.spec->alias;
    if (!StringUtil::CIEquals(alias, expected_alias)) {
        return false;
    }
    auto replacement = Subquery(plan.view->Copy(), alias);
    replacement->sample = std::move(relation->sample);
    replacement->column_name_alias = std::move(relation->column_name_alias);
    relation = std::move(replacement);
    return true;
}

class WindowRewrite {
public:
    WindowRewrite(SelectNode &owner_p, vector<SourcePlan> &sources_p,
                  const vector<MeasureWindowCall> &calls_p, const ParserOptions &options_p,
                  const Names &input_columns_p)
        : owner(owner_p), sources(sources_p), calls(calls_p), options(options_p), grouped(IsGrouped(owner_p)),
          aliases(Aliases(owner_p)), input_columns(input_columns_p) {
    }

    unique_ptr<QueryNode> Run() {
        auto output = make_uniq<SelectNode>();
        auto projections = std::move(owner.select_list);
        auto qualify = std::move(owner.qualify);
        output->modifiers = std::move(owner.modifiers);

        if (owner.aggregate_handling == AggregateHandling::FORCE_AGGREGATES) {
            // DuckDB's GROUP BY ALL binder cannot infer groups from the new
            // list window. Infer them from the user's original projection.
            owner.aggregate_handling = AggregateHandling::STANDARD_HANDLING;
            owner.groups = GroupByNode();
            GroupingSet grouping;
            for (auto &projection : projections) {
                bool window = false;
                Walk(projection, [&](unique_ptr<ParsedExpression> &node) {
                    window |= node->GetExpressionClass() == ExpressionClass::WINDOW;
                    return !window;
                });
                if (window || HasAggregate(projection)) {
                    continue;
                }
                auto group = projection->Copy();
                group->ClearAlias();
                grouping.insert(ProjectionIndex(owner.groups.group_expressions.size()));
                owner.groups.group_expressions.push_back(std::move(group));
            }
            owner.groups.grouping_sets.push_back(std::move(grouping));
        }

        for (auto &group : owner.groups.group_expressions) {
            // SQL ordinal grouping is relative to the original projection.
            if (group->GetExpressionClass() == ExpressionClass::CONSTANT) {
                auto text = group->ToString();
                if (!text.empty() && text.find_first_not_of("0123456789") == string::npos) {
                    auto ordinal = std::stoull(text);
                    if (ordinal > 0 && ordinal <= projections.size()) {
                        group = projections[ordinal - 1]->Copy();
                        group->ClearAlias();
                    }
                }
            }
            ExpandAliases(group, aliases, input_columns);
        }
        if (owner.having) {
            ExpandAliases(owner.having, aliases, input_columns);
        }
        if (owner.where_clause) {
            ExpandAliases(owner.where_clause, aliases, input_columns);
        }
        for (auto &projection : projections) {
            auto name = projection->GetName();
            projection->ClearAlias();
            ExpandAliases(projection, aliases, input_columns);
            Rewrite(projection);
            projection->SetAlias(name);
            output->select_list.push_back(std::move(projection));
        }
        if (qualify) {
            ExpandAliases(qualify, aliases, input_columns);
            Rewrite(qualify);
            output->where_clause = std::move(qualify);
        }
        for (auto &modifier : output->modifiers) {
            if (modifier->type == ResultModifierType::ORDER_MODIFIER) {
                for (auto &order : modifier->Cast<OrderModifier>().orders) {
                    RewriteModifier(order.expression);
                }
            } else if (modifier->type == ResultModifierType::DISTINCT_MODIFIER) {
                for (auto &target : modifier->Cast<DistinctModifier>().distinct_on_targets) {
                    RewriteModifier(target);
                }
            }
        }
        output->cte_map = std::move(owner.cte_map);
        output->from_table = Subquery(owner.Copy(), "__ys_window_stage");
        return std::move(output);
    }

private:
    const MeasureWindowCall *Call(const ParsedExpression &expression) const {
        if (expression.GetExpressionClass() != ExpressionClass::WINDOW) {
            return nullptr;
        }
        auto &window = expression.Cast<WindowExpression>();
        for (auto &call : calls) {
            if (StringUtil::CIEquals(window.FunctionName().GetIdentifierName(), call.marker_name)) {
                return &call;
            }
        }
        return nullptr;
    }

    bool ContainsCall(unique_ptr<ParsedExpression> &expression) const {
        bool found = false;
        Walk(expression, [&](unique_ptr<ParsedExpression> &node) {
            found = found || Call(*node);
            return !found;
        });
        return found;
    }

    string Project(unique_ptr<ParsedExpression> expression) {
        auto name = "__ys_window_value_" + std::to_string(owner.select_list.size());
        expression->SetAlias(Identifier(name));
        owner.select_list.push_back(std::move(expression));
        return name;
    }

    void RewriteModifier(unique_ptr<ParsedExpression> &expression) {
        // Output aliases and ordinal references are resolved by the final query.
        if (expression->GetExpressionClass() == ExpressionClass::CONSTANT) {
            return;
        }
        if (expression->GetExpressionClass() == ExpressionClass::COLUMN_REF) {
            auto &column = expression->Cast<ColumnRefExpression>();
            if (!column.IsQualified() && aliases.count(Key(column.GetColumnName().GetIdentifierName()))) {
                return;
            }
        }
        ExpandAliases(expression, aliases, input_columns);
        Rewrite(expression);
    }

    string DimensionSQL(const SourcePlan &source, const string &dimension, const string &qualifier) const {
        auto expression = Expression(dimension, options);
        if (expression->GetExpressionClass() == ExpressionClass::COLUMN_REF) {
            auto key = Key(expression->Cast<ColumnRefExpression>().GetColumnName().GetIdentifierName());
            auto match = source.dimensions.find(key);
            if (match != source.dimensions.end()) {
                expression = match->second->Copy();
            }
        }
        RebindColumns(expression, source, qualifier);
        return expression->ToString();
    }

    string Context(const SourcePlan &source, const MeasureWindowCall &call, const string &frame,
                   const string &visible_frame, const string &payload_ids = "") const {
        const string candidate = "__ys_candidate";
        const string selected = "__ys_selected";
        auto frame_ids = payload_ids.empty() ? "SELECT unnest(flatten(" + Quote(frame) + "))" : payload_ids;
        auto membership = candidate + "." + Quote(source.id_name) + " IN (" + frame_ids + ")";
        std::unordered_set<string> removed;
        struct SetOverride {
            string dimension;
            string value;
        };
        std::unordered_map<string, SetOverride> sets;
        string condition;
        bool global = false;
        bool expand = call.modifiers.size() > 1;
        bool has_set = false;
        for (auto &modifier : call.modifiers) {
            has_set |= modifier.type == WindowContextType::SET;
        }
        bool visible = false;
        auto dimension_key = [&](const string &dimension) {
            auto expression = Expression(dimension, options);
            if (expression->GetExpressionClass() == ExpressionClass::COLUMN_REF) {
                return Key(expression->Cast<ColumnRefExpression>().GetColumnName().GetIdentifierName());
            }
            return Key(expression->ToString());
        };
        auto resolve_current = [&](unique_ptr<ParsedExpression> &expression, bool bare_dimensions) {
            Walk(expression, [&](unique_ptr<ParsedExpression> &node) {
                string dimension;
                if (node->GetExpressionClass() == ExpressionClass::FUNCTION) {
                    auto &function = node->Cast<FunctionExpression>();
                    if (StringUtil::CIEquals(function.FunctionName().GetIdentifierName(), "current") &&
                        function.GetArguments().size() == 1) {
                        dimension = function.GetArguments()[0].GetExpression().ToString();
                    }
                } else if (bare_dimensions && node->GetExpressionClass() == ExpressionClass::COLUMN_REF) {
                    auto &column = node->Cast<ColumnRefExpression>();
                    if (source.dimensions.count(Key(column.GetColumnName().GetIdentifierName()))) {
                        dimension = node->ToString();
                    }
                }
                if (dimension.empty()) {
                    return true;
                }
                auto dim = DimensionSQL(source, dimension, selected);
                node = Expression("(SELECT CASE WHEN count(DISTINCT " + dim +
                                  ") + max(CASE WHEN " + dim + " IS NULL THEN 1 ELSE 0 END) = 1 THEN first(" + dim +
                                  ") ELSE NULL END FROM " + Quote(source.base_name) + " " + selected + " WHERE " +
                                  selected + "." + Quote(source.id_name) + " IN (" + frame_ids + "))", options);
                return false;
            });
        };
        for (auto it = call.modifiers.rbegin(); it != call.modifiers.rend(); ++it) {
            auto &modifier = *it;
            switch (modifier.type) {
            case WindowContextType::ALL_GLOBAL:
                global = true;
                condition.clear();
                visible = false;
                removed.clear();
                sets.clear();
                break;
            case WindowContextType::ALL:
                removed.insert(dimension_key(modifier.dimension));
                expand = true;
                break;
            case WindowContextType::SET:
                if (!global && !removed.count(dimension_key(modifier.dimension))) {
                    sets[dimension_key(modifier.dimension)] = {modifier.dimension, modifier.value};
                    expand = true;
                }
                break;
            case WindowContextType::WHERE:
                if (!global) {
                    auto expression = ModifierExpression(modifier.value, options);
                    NormalizeModifierReferences(expression, source);
                    RebindColumns(expression, source, candidate);
                    resolve_current(expression, false);
                    condition = expression->ToString();
                    visible = false;
                }
                break;
            case WindowContextType::VISIBLE:
                if (!global && !has_set) {
                    condition.clear();
                    visible = true;
                }
                break;
            }
        }
        if (global) {
            return "true";
        }
        if (!condition.empty() && call.modifiers.size() == 1) {
            return condition;
        }
        if (!expand) {
            return membership;
        }
        vector<string> correlations;
        for (auto &dimension : source.dimensions) {
            auto base_expression = dimension.second->ToString();
            bool replaced = false;
            for (auto &entry : sets) {
                replaced |= entry.first == dimension_key(dimension.first) ||
                            entry.first == dimension_key(base_expression);
            }
            if (!removed.count(dimension_key(dimension.first)) &&
                !removed.count(dimension_key(base_expression)) && !replaced) {
                correlations.push_back(DimensionSQL(source, dimension.first, candidate) +
                                       " IS NOT DISTINCT FROM " + DimensionSQL(source, dimension.first, selected));
            }
        }
        bool needs_selected_row = !correlations.empty();
        for (auto &entry : sets) {
            auto value = ModifierExpression(entry.second.value, options);
            // CURRENT values use the single value of the frame dimension. A
            // multi-valued or empty frame has NULL as its current value.
            resolve_current(value, true);
            correlations.push_back(DimensionSQL(source, entry.second.dimension, candidate) + " IS NOT DISTINCT FROM " +
                                   value->ToString());
        }
        if (!condition.empty()) {
            correlations.push_back(condition);
        }
        if (visible && !visible_frame.empty()) {
            correlations.push_back(candidate + "." + Quote(source.id_name) +
                                   " IN (SELECT unnest(flatten(" + Quote(visible_frame) + ")))");
        }
        if (!needs_selected_row) {
            return correlations.empty() ? "true" : StringUtil::Join(correlations, " AND ");
        }
        correlations.push_back(selected + "." + Quote(source.id_name) + " IN (" + frame_ids + ")");
        return "EXISTS (SELECT 1 FROM " + Quote(source.base_name) + " " + selected + " WHERE " +
               StringUtil::Join(correlations, " AND ") + ")";
    }

    void Rewrite(unique_ptr<ParsedExpression> &expression) {
        if (auto call = Call(*expression)) {
            SourcePlan *source = nullptr;
            for (auto &candidate : sources) {
                if (candidate.spec->key == call->source_key) {
                    source = &candidate;
                    break;
                }
            }
            if (!source) {
                throw InternalException("Missing source for measure window %s", call->marker_name);
            }
            auto window = expression->Copy();
            auto &list = window->Cast<WindowExpression>();
            auto measure = Expression(call->expression_sql, options);
            const auto order_count = list.ArgOrders().size();
            vector<unique_ptr<ParsedExpression>> caller_keys;
            Walk(measure, [&](unique_ptr<ParsedExpression> &node) {
                if (!caller_keys.empty() || node->GetExpressionClass() != ExpressionClass::FUNCTION) {
                    return caller_keys.empty();
                }
                auto &function = node->Cast<FunctionExpression>();
                if (order_count && function.OrderBy() && function.OrderBy()->orders.size() >= order_count) {
                    for (idx_t i = 0; i < order_count; i++) {
                        caller_keys.push_back(function.OrderBy()->orders[i].expression->Copy());
                    }
                    return false;
                }
                return true;
            });
            vector<bool> external_keys;
            vector<string> payload_fields;
            bool external_order = false;
            for (idx_t i = 0; i < caller_keys.size(); i++) {
                bool external = false;
                CapturedReferences references(source->qualifiers, [&](unique_ptr<ParsedExpression> &node) {
                    if (IsSourceReference(*node, *source, true)) {
                        return;
                    }
                    external = true;
                    auto field = "__ys_ext_" + std::to_string(payload_fields.size());
                    auto input = node->Copy();
                    ExpandAliases(input, aliases, input_columns);
                    payload_fields.push_back(Quote(field) + " := " + input->ToString());
                    node = Expression("__ys_payload." + Quote(field), options);
                });
                // Grouped consumer keys are evaluated at the window input grain.
                // Source-only keys instead remain expressions over original rows.
                auto raw_key = list.ArgOrders()[i].expression->Copy();
                ExpandAliases(raw_key, aliases, input_columns);
                if (HasAggregate(raw_key)) {
                    external = true;
                    auto field = "__ys_ext_" + std::to_string(payload_fields.size());
                    payload_fields.push_back(Quote(field) + " := " + raw_key->ToString());
                    caller_keys[i] = Expression("__ys_payload." + Quote(field), options);
                } else {
                    references.Expression(caller_keys[i]);
                }
                external_keys.push_back(external);
                external_order |= external;
            }
            vector<string> occurrence_orders;
            for (idx_t i = 0; i < caller_keys.size(); i++) {
                auto &original = list.ArgOrders()[i];
                OrderByNode order(original.type, original.null_order,
                                  Expression("__ys_key_" + std::to_string(i), options));
                occurrence_orders.push_back(order.ToString());
            }
            list.SetFunctionName("list");
            list.DistinctMutable() = false;
            list.ArgOrdersMutable().clear();
            list.GetArgumentsMutable().clear();
            auto alias = source->spec->alias.empty() ? source->spec->relation_name : source->spec->alias;
            auto lineage_sql = Quote(alias) + "." + Quote(source->lineage_name);
            if (grouped) {
                lineage_sql = "flatten(list(" + lineage_sql + "))";
            }
            auto frame_value = lineage_sql;
            if (external_order) {
                frame_value = "struct_pack(ids := " + lineage_sql + ", " +
                              StringUtil::Join(payload_fields, ", ") + ")";
            }
            list.GetArgumentsMutable().emplace_back(Expression(frame_value, options));
            if (list.Filter()) {
                ExpandAliases(list.FilterMutable(), aliases, input_columns);
            }
            for (auto &partition : list.PartitionsMutable()) {
                ExpandAliases(partition, aliases, input_columns);
            }
            for (auto &order : list.OrderByMutable()) {
                ExpandAliases(order.expression, aliases, input_columns);
            }
            if (list.StartExpr()) {
                ExpandAliases(list.StartExprMutable(), aliases, input_columns);
            }
            if (list.EndExpr()) {
                ExpandAliases(list.EndExprMutable(), aliases, input_columns);
            }
            auto frame = Project(std::move(window));
            string visible_frame;
            for (auto &modifier : call->modifiers) {
                if (modifier.type == WindowContextType::VISIBLE) {
                    auto existing = visible_frames.find(source->spec->key);
                    if (existing == visible_frames.end()) {
                        visible_frame = Project(Expression("list(" + lineage_sql + ") OVER ()", options));
                        visible_frames.emplace(source->spec->key, visible_frame);
                    } else {
                        visible_frame = existing->second;
                    }
                    break;
                }
            }
            string order_cte;
            string order_join;
            string frame_ids;
            if (external_order) {
                frame_ids = "SELECT unnest(__ys_payload.ids) FROM unnest(" + Quote(frame) +
                            ") __ys_frames(__ys_payload)";
                vector<string> key_projections;
                for (idx_t i = 0; i < caller_keys.size(); i++) {
                    RebindColumns(caller_keys[i], *source, "__ys_candidate");
                    key_projections.push_back(caller_keys[i]->ToString() + " AS __ys_key_" + std::to_string(i));
                }
                order_cte = "WITH __ys_occurrences AS MATERIALIZED (SELECT __ys_candidate." +
                            Quote(source->id_name) + ", " + StringUtil::Join(key_projections, ", ") +
                            " FROM unnest(" + Quote(frame) + ") __ys_frames(__ys_payload), " +
                            "unnest(__ys_payload.ids) __ys_ids(id) JOIN " + Quote(source->base_name) +
                            " __ys_candidate ON __ys_candidate." + Quote(source->id_name) + " = __ys_ids.id), " +
                            "__ys_order_keys AS (SELECT * FROM __ys_occurrences QUALIFY row_number() OVER " +
                            "(PARTITION BY " + Quote(source->id_name) + " ORDER BY " +
                            StringUtil::Join(occurrence_orders, ", ") + ") = 1) ";
                // Context expansion can introduce rows without frame occurrences.
                // Their joined-input keys are NULL; source keys still recompute.
                order_join = " LEFT JOIN __ys_order_keys USING (" + Quote(source->id_name) + ")";
                Walk(measure, [&](unique_ptr<ParsedExpression> &node) {
                    if (node->GetExpressionClass() != ExpressionClass::FUNCTION) {
                        return true;
                    }
                    auto &function = node->Cast<FunctionExpression>();
                    if (function.OrderBy() && function.OrderBy()->orders.size() >= order_count) {
                        for (idx_t i = 0; i < order_count; i++) {
                            if (external_keys[i]) {
                                function.OrderByMutable()->orders[i].expression =
                                    Expression("__ys_order_keys.__ys_key_" + std::to_string(i), options);
                            }
                        }
                    }
                    return true;
                });
            }
            RebindColumns(measure, *source, "__ys_candidate");
            expression = Expression("(" + order_cte + "SELECT " + measure->ToString() + " FROM " +
                                    Quote(source->base_name) + " __ys_candidate" + order_join + " WHERE " +
                                    Context(*source, *call, frame, visible_frame, frame_ids) + ")", options);
            return;
        }
        if (!ContainsCall(expression)) {
            auto name = Project(std::move(expression));
            expression = make_uniq<ColumnRefExpression>(Identifier(name));
            return;
        }
        ParsedExpressionIterator::EnumerateChildren(*expression, [&](unique_ptr<ParsedExpression> &child) {
            Rewrite(child);
        });
    }

    SelectNode &owner;
    vector<SourcePlan> &sources;
    const vector<MeasureWindowCall> &calls;
    const ParserOptions &options;
    bool grouped;
    ExpressionMap aliases;
    Names input_columns;
    std::unordered_map<string, string> visible_frames;
};

} // namespace

string RewriteNativeMeasureWindows(const string &scope_sql, const vector<MeasureWindowSource> &sources,
                                   const vector<MeasureWindowCall> &calls, const vector<string> &visible_ctes,
                                   const ParserOptions &options) {
    if (calls.empty()) {
        return scope_sql;
    }
    auto query = Query(scope_sql, options);
    auto &owner = query->Cast<SelectNode>();
    ExpandStars(owner);
    auto input_columns = InputColumns(owner);
    std::unordered_map<string, idx_t> caller_order_counts;
    for (auto &call : calls) {
        caller_order_counts.emplace(call.marker_name, 0);
    }
    SelectExpressions(owner, [&](unique_ptr<ParsedExpression> &expression) {
        Walk(expression, [&](unique_ptr<ParsedExpression> &node) {
            if (node->GetExpressionClass() == ExpressionClass::WINDOW) {
                auto &window = node->Cast<WindowExpression>();
                auto entry = caller_order_counts.find(window.FunctionName().GetIdentifierName());
                if (entry != caller_order_counts.end()) {
                    entry->second = window.ArgOrders().size();
                }
            }
            return true;
        });
    });
    vector<SourcePlan> plans;
    Names cte_names;
    for (auto &name : visible_ctes) {
        cte_names.insert(Key(name));
    }
    for (auto &entry : owner.cte_map.map) {
        cte_names.insert(Key(entry.first.GetIdentifierName()));
    }
    for (idx_t i = 0; i < sources.size(); i++) {
        plans.push_back(BuildSource(sources[i], calls, caller_order_counts, i, cte_names, options));
    }
    for (auto &plan : plans) {
        if (!ReplaceSource(owner.from_table, plan)) {
            throw BinderException("Cannot locate measure window source %s in the query", plan.spec->relation_name);
        }
    }
    WindowRewrite rewrite(owner, plans, calls, options, input_columns);
    auto output = rewrite.Run();
    for (auto &plan : plans) {
        auto cte = make_uniq<CommonTableExpressionInfo>();
        cte->query_node = std::move(plan.base);
        cte->materialized = CTEMaterialize::CTE_MATERIALIZE_ALWAYS;
        output->cte_map.map.insert(Identifier(plan.base_name), std::move(cte));
    }
    if (!visible_ctes.empty()) {
        // This SELECT is spliced into a statement with an enclosing WITH.
        // Keep the generated WITH inside a subquery rather than emitting two
        // adjacent WITH clauses at the same query level.
        auto wrapper = make_uniq<SelectNode>();
        wrapper->select_list.push_back(make_uniq<StarExpression>());
        wrapper->from_table = Subquery(std::move(output), "__ys_window_result");
        return wrapper->ToString();
    }
    return output->ToString();
}

} // namespace duckdb
#endif
