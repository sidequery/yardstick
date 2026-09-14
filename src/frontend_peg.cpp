#include "frontend_peg.hpp"

#if YARDSTICK_GRAMMAR_EXTENSION
#include "native_statement_traversal.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/parser/expression/columnref_expression.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/parser/expression/subquery_expression.hpp"
#include "duckdb/parser/expression/window_expression.hpp"
#include "duckdb/parser/parsed_expression_iterator.hpp"
#include "duckdb/parser/grammar_extension.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/parser/parsed_data/create_view_info.hpp"
#include "duckdb/parser/query_node/select_node.hpp"
#include "duckdb/parser/statement/create_statement.hpp"
#include "duckdb/parser/peg/compiled_grammar.hpp"
#include "duckdb/parser/peg/transformer/peg_transformer.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/table_binding.hpp"
#include "duckdb/parser/statement/select_statement.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>

namespace duckdb {
namespace {

struct SyntaxSpan {
    idx_t start;
    idx_t end;
    bool measure;
};

struct NativeModifier {
    YardstickAtType type;
    string dimension;
    string value;
};

struct NativeAtClause {
    idx_t start;
    idx_t end;
    vector<NativeModifier> modifiers;
    const ParsedExpression *marker;
};

struct SyntaxCapture {
    struct CurrentReference {
        string dimension;
        idx_t start;
        idx_t end;
    };
    vector<CurrentReference> current_references;
    vector<SyntaxSpan> spans;
    bool has_measure = false;
    const string *source = nullptr;
    vector<NativeAtClause> clauses;
    vector<unique_ptr<ParsedExpression>> modifier_expressions;
    struct Measure {
        string expression;
        string name;
        string alias;
        idx_t start;
        idx_t end;
        const ParsedExpression *marker;
    };
    vector<Measure> measures;
};

// Grammar callbacks have no per-parse extension state. This scope captures only
// source syntax, never catalog state, and restores nested parses on exit.
thread_local SyntaxCapture *active_capture = nullptr;
thread_local const NativeYardstickParseScope *active_parse_scope = nullptr;
thread_local ClientContext *active_bind_context = nullptr;

// CURRENT is an expression only inside an AT SET value or WHERE predicate.
// Work from native tokens so literals, quoted names and comments cannot open
// a scope, and a nested SELECT starts a separate SQL expression namespace.
class CurrentKeywordMatcher final : public AtomicMatcher {
public:
    CurrentKeywordMatcher() : AtomicMatcher(MatcherType::CUSTOM) {}

    MatcherResult MatchAtomic(MatchState &state) const override {
        auto token = state.token_iterator.Current();
        if (!token) {
            return MatcherResult::Failure();
        }
        bool keyword = StringUtil::CIEquals(token->text, "CURRENT");
        if (!keyword && StringUtil::CIEquals(token->text, "\"CURRENT\"")) {
            auto next = state.token_iterator.Position() + 1;
            while (next < state.token_iterator.Size() &&
                   state.token_iterator.GetToken(next).type == TokenType::COMMENT) {
                next++;
            }
            // DuckDB quotes the canonical function name when rendering it.
            // A quoted column named current must still retain alias syntax.
            keyword = next < state.token_iterator.Size() && state.token_iterator.GetToken(next).text == "(";
        }
        if (!keyword) {
            return MatcherResult::Failure();
        }
        struct Frame {
            bool at = false;
            bool value = false;
            bool set = false;
            bool query = false;
            bool started = false;
        };
        vector<Frame> frames(1);
        const MatcherToken *previous = nullptr;
        for (idx_t i = 0; i < state.token_iterator.Position(); i++) {
            auto &item = state.token_iterator.GetToken(i);
            if (item.type == TokenType::COMMENT) {
                continue;
            }
            if (item.text == "(") {
                Frame frame;
                frame.at = previous && StringUtil::CIEquals(previous->text, "AT");
                frames.back().started = true;
                frames.push_back(frame);
            } else if (item.text == ")") {
                if (frames.size() > 1) {
                    frames.pop_back();
                }
            } else if (StringUtil::CIEquals(item.text, "SELECT") ||
                       (!frames.back().started &&
                        (StringUtil::CIEquals(item.text, "VALUES") || StringUtil::CIEquals(item.text, "FROM") ||
                         StringUtil::CIEquals(item.text, "WITH") || StringUtil::CIEquals(item.text, "TABLE") ||
                         StringUtil::CIEquals(item.text, "PIVOT") || StringUtil::CIEquals(item.text, "UNPIVOT")))) {
                frames.back().query = true;
                frames.back().started = true;
            } else if (frames.back().at) {
                if (StringUtil::CIEquals(item.text, "SET")) {
                    frames.back().set = true;
                    frames.back().value = false;
                } else if (StringUtil::CIEquals(item.text, "WHERE")) {
                    frames.back().set = false;
                    frames.back().value = true;
                } else if (frames.back().set && item.text == "=") {
                    frames.back().value = true;
                } else if (StringUtil::CIEquals(item.text, "ALL") ||
                           StringUtil::CIEquals(item.text, "VISIBLE")) {
                    frames.back().set = false;
                    frames.back().value = false;
                }
                frames.back().started = true;
            } else {
                frames.back().started = true;
            }
            previous = &item;
        }
        bool allowed = false;
        for (auto frame = frames.rbegin(); frame != frames.rend(); ++frame) {
            if (frame->query) {
                break;
            }
            if (frame->at) {
                allowed = frame->value;
                break;
            }
        }
        if (!allowed) {
            return MatcherResult::Failure();
        }
        auto result = state.AllocateParseResult<KeywordParseResult>(token->text, token->offset, token->length);
        state.token_iterator.Advance();
        state.UpdateMaxTokenIndex();
        return result;
    }

    SuggestionType AddSuggestionInternal(MatchState &) const override {
        return SuggestionType::OPTIONAL;
    }
    string ToString() const override { return "CURRENT in AT expression"; }
};

struct CaptureScope {
    explicit CaptureScope(SyntaxCapture &capture) : previous(active_capture) {
        active_capture = &capture;
    }
    ~CaptureScope() { active_capture = previous; }
    SyntaxCapture *previous;
};

void CaptureSpan(ParseResult &first, ParseResult &last, bool measure = false) {
    if (!active_capture) {
        throw ParserException("Yardstick syntax must be processed by the Yardstick parser override");
    }
    active_capture->spans.push_back({first.offset.GetIndex(),
                                    last.offset.GetIndex() + last.length.GetIndex(), measure});
}

ParseResult &UnwrapModifier(ParseResult &result) {
    auto *current = &result;
    while (true) {
        if (current->type == ParseResultType::CHOICE) {
            current = &current->Cast<ChoiceParseResult>().GetResult();
        } else if (current->type == ParseResultType::LIST &&
                   current->Cast<ListParseResult>().GetChildren().size() == 1) {
            // Named rules wrap non-sequence bodies in a singleton list.
            current = &current->Cast<ListParseResult>().GetChild(0);
        } else {
            return *current;
        }
    }
}

string RenderExpression(PEGTransformer &transformer, ParseResult &result) {
    // Resolve syntax supplied by every active grammar before shared lowering.
    auto expression = transformer.Transform<unique_ptr<ParsedExpression>>(result);
    auto sql = expression->ToString();
    // AT markers retain only their operand in the main AST. Keep modifier
    // expressions alive for validation before rendering loses their structure.
    active_capture->modifier_expressions.push_back(std::move(expression));
    return sql;
}

string DimensionSource(ParseResult &result) {
    // Dimension matching shares the outer query's spelling. Keep its source
    // expression, with boundaries supplied by PEG rather than a text scanner.
    auto &sql = *active_capture->source;
    auto start = result.offset.GetIndex();
    auto length = result.length.GetIndex();
    if (start > sql.size() || length > sql.size() - start) {
        throw ParserException("Yardstick dimension source location is out of range");
    }
    return sql.substr(start, length);
}

string ModifierKeyword(ParseResult &result) {
    auto &unwrapped = UnwrapModifier(result);
    ParseResult *keyword = &unwrapped;
    if (unwrapped.type == ParseResultType::LIST) {
        auto &list = unwrapped.Cast<ListParseResult>();
        if (list.GetChildren().empty()) {
            return string();
        }
        keyword = &UnwrapModifier(list.GetChild(0));
    }
    if (keyword->type != ParseResultType::KEYWORD) {
        return string();
    }
    return StringUtil::Upper(keyword->Cast<KeywordParseResult>().keyword);
}

void ReadModifier(PEGTransformer &transformer, ParseResult &result, vector<NativeModifier> &modifiers);

void ReadAllDimensions(PEGTransformer &transformer, ParseResult &result, vector<NativeModifier> &modifiers) {
    auto &all = UnwrapModifier(result).Cast<ListParseResult>();
    auto &optional_tail = all.Child<OptionalParseResult>(1);
    if (!optional_tail.HasResult()) {
        modifiers.push_back({YARDSTICK_AT_ALL_GLOBAL, string(), string()});
        return;
    }
    auto *tail = &optional_tail.GetResult();
    bool has_dimension = false;
    while (tail) {
        auto &node = UnwrapModifier(*tail);
        auto keyword = ModifierKeyword(node);
        if (keyword == "ALL" || keyword == "SET" || keyword == "WHERE" || keyword == "VISIBLE") {
            if (!has_dimension) {
                modifiers.push_back({YARDSTICK_AT_ALL_GLOBAL, string(), string()});
            }
            ReadModifier(transformer, node, modifiers);
            return;
        }
        auto &dimension_tail = node.Cast<ListParseResult>();
        modifiers.push_back({YARDSTICK_AT_ALL_DIM, DimensionSource(dimension_tail.GetChild(0)), string()});
        has_dimension = true;
        auto &next = dimension_tail.Child<OptionalParseResult>(1);
        tail = next.HasResult() ? &next.GetResult() : nullptr;
    }
}

void ReadModifier(PEGTransformer &transformer, ParseResult &result, vector<NativeModifier> &modifiers) {
    auto &node = UnwrapModifier(result);
    auto keyword = ModifierKeyword(node);
    if (keyword == "VISIBLE") {
        modifiers.push_back({YARDSTICK_AT_VISIBLE, string(), string()});
    } else if (keyword == "ALL") {
        ReadAllDimensions(transformer, node, modifiers);
    } else if (keyword == "SET") {
        auto &list = node.Cast<ListParseResult>();
        modifiers.push_back({YARDSTICK_AT_SET, DimensionSource(list.GetChild(1)),
                             RenderExpression(transformer, list.GetChild(3))});
    } else if (keyword == "WHERE") {
        auto &list = node.Cast<ListParseResult>();
        modifiers.push_back({YARDSTICK_AT_WHERE, string(), RenderExpression(transformer, list.GetChild(1))});
    } else {
        throw ParserException("Unsupported Yardstick modifier parse result");
    }
}

unique_ptr<TransformResultValue> TransformAt(PEGTransformer &transformer, ParseResult &result) {
    CaptureSpan(result, result);
    // BaseExpression inserts the left operand into this marker. Keep its
    // identity so aggregate traversal associates suffixes through AST parents.
    unique_ptr<ParsedExpression> expression =
        make_uniq<FunctionExpression>("__yardstick_at", vector<unique_ptr<ParsedExpression>> {});
    if (active_capture->source) {
        NativeAtClause clause {result.offset.GetIndex(), result.offset.GetIndex() + result.length.GetIndex(), {},
                               expression.get()};
        auto &list = result.Cast<ListParseResult>();
        for (auto &child : list.Child<RepeatParseResult>(2).GetChildren()) {
            ReadModifier(transformer, child.get(), clause.modifiers);
        }
        active_capture->clauses.push_back(std::move(clause));
    }
    return make_uniq<TypedTransformResult<unique_ptr<ParsedExpression>>>(std::move(expression));
}

unique_ptr<TransformResultValue> TransformMeasure(PEGTransformer &transformer, ParseResult &result) {
    auto &list = result.Cast<ListParseResult>();
    vector<unique_ptr<ParsedExpression>> children;
    children.push_back(transformer.Transform<unique_ptr<ParsedExpression>>(list.GetChild(0)));
    unique_ptr<ParsedExpression> expression = make_uniq<FunctionExpression>("__yardstick_measure", std::move(children));
    expression->SetAlias(transformer.Transform<Identifier>(list.GetChild(3)));
    CaptureSpan(list.GetChild(1), list.GetChild(3), true);
    if (active_capture) {
        active_capture->has_measure = true;
        if (active_capture->source) {
            auto &source_expression = list.GetChild(0);
            auto &source_alias = list.GetChild(3);
            active_capture->measures.push_back({DimensionSource(source_expression),
                                               expression->GetAlias().GetIdentifierName(), DimensionSource(source_alias),
                                               source_expression.offset.GetIndex(),
                                               source_alias.offset.GetIndex() + source_alias.length.GetIndex(),
                                               expression.get()});
        }
    }
    return make_uniq<TypedTransformResult<unique_ptr<ParsedExpression>>>(std::move(expression));
}

unique_ptr<TransformProcess> StartAt(PEGTransformer &transformer, ParseResult &result) {
    return make_uniq<FinalizeTransformProcess>(transformer, result, TransformAt);
}

unique_ptr<TransformResultValue> TransformCurrentDimension(PEGTransformer &transformer, ParseResult &result,
                                                         idx_t dimension_index) {
    auto &list = result.Cast<ListParseResult>();
    vector<unique_ptr<ParsedExpression>> children;
    children.push_back(transformer.Transform<unique_ptr<ParsedExpression>>(list.GetChild(dimension_index)));
    if (active_capture) {
        active_capture->current_references.push_back({children[0]->ToString(), result.offset.GetIndex(),
                                                      result.offset.GetIndex() + result.length.GetIndex()});
    }
    unique_ptr<ParsedExpression> expression = make_uniq<FunctionExpression>("current", std::move(children));
    return make_uniq<TypedTransformResult<unique_ptr<ParsedExpression>>>(std::move(expression));
}

unique_ptr<TransformProcess> StartCurrentReference(PEGTransformer &transformer, ParseResult &result) {
    return make_uniq<FinalizeTransformProcess>(transformer, result,
        [](PEGTransformer &transformer, ParseResult &result) {
            return TransformCurrentDimension(transformer, result, 1);
        });
}

unique_ptr<TransformProcess> StartCurrentCall(PEGTransformer &transformer, ParseResult &result) {
    return make_uniq<FinalizeTransformProcess>(transformer, result,
        [](PEGTransformer &transformer, ParseResult &result) {
            return TransformCurrentDimension(transformer, result, 2);
        });
}

unique_ptr<TransformProcess> StartMeasure(PEGTransformer &transformer, ParseResult &result) {
    return make_uniq<FinalizeTransformProcess>(transformer, result, TransformMeasure);
}

bool ExpandDeclarationReferences(unique_ptr<ParsedExpression> &expression,
                                 const vector<SyntaxCapture::Measure> &declarations,
                                 vector<idx_t> &active) {
    if (expression->GetExpressionClass() == ExpressionClass::SUBQUERY ||
        expression->GetExpressionClass() == ExpressionClass::WINDOW) {
        return false;
    }
    const ParsedExpression *reference = expression.get();
    if (expression->GetExpressionClass() == ExpressionClass::FUNCTION) {
        auto &function = expression->Cast<FunctionExpression>();
        auto name = function.FunctionName().GetIdentifierName();
        if (StringUtil::CIEquals(name, "aggregate") && function.GetArguments().size() == 1) {
            reference = &function.GetArguments()[0].GetExpression();
        } else if (IsYardstickStandardAggregate(name)) {
            // These identifiers belong to base rows, not sibling declarations.
            return false;
        }
    }
    if (reference->GetExpressionClass() == ExpressionClass::COLUMN_REF) {
        auto &names = reference->Cast<ColumnRefExpression>().ColumnNames();
        if (names.size() == 1) {
            for (idx_t i = 0; i < declarations.size(); i++) {
                auto &declaration = declarations[i];
                if (names[0] != declaration.marker->GetAlias()) {
                    continue;
                }
                if (std::find(active.begin(), active.end(), i) != active.end()) {
                    throw ParserException("Cyclic AS MEASURE declaration reference");
                }
                auto replacement = declaration.marker->Cast<FunctionExpression>()
                    .GetArguments()[0].GetExpression().Copy();
                active.push_back(i);
                ExpandDeclarationReferences(replacement, declarations, active);
                active.pop_back();
                expression = std::move(replacement);
                return true;
            }
        }
    }
    bool changed = false;
    ParsedExpressionIterator::EnumerateChildren(*expression, [&](unique_ptr<ParsedExpression> &child) {
        changed |= ExpandDeclarationReferences(child, declarations, active);
    });
    return changed;
}

class YardstickGrammar final : public GrammarExtension {
public:
    YardstickGrammar() : GrammarExtension("yardstick", "Yardstick measure aliases and context modifiers") {}

    vector<GrammarChange> GetChanges() const override {
        return {
            GrammarChange::AddRule("YardstickCurrentKeyword <- 'CURRENT'"),
            GrammarChange::AddTerminalRuleOverride("YardstickCurrentKeyword", [](const PEGKeywordHelper &) {
                return make_uniq<CurrentKeywordMatcher>();
            }),
            GrammarChange::AddRule("YardstickCurrentReference <- YardstickCurrentKeyword ColumnReference", StartCurrentReference),
            GrammarChange::AddRule("YardstickCurrentCall <- YardstickCurrentKeyword '(' ColumnReference ')'", StartCurrentCall),
            GrammarChange::PrependChoice("SingleExpression", "YardstickCurrentReference"),
            GrammarChange::PrependChoice("SingleExpression", "YardstickCurrentCall"),
            GrammarChange::AddRule("YardstickAtModifier <- 'AT' '(' YardstickAtClause+ ')'", StartAt),
            GrammarChange::AddRule("YardstickAtClause <- YardstickAtAll / YardstickAtSet / YardstickAtWhere / 'VISIBLE'"),
            GrammarChange::AddRule("YardstickAtAll <- 'ALL' YardstickAtAllTail?"),
            GrammarChange::AddRule("YardstickAtAllTail <- YardstickAtClause / BaseExpression YardstickAtAllTail?"),
            GrammarChange::AddRule("YardstickAtSet <- 'SET' BaseExpression '=' Expression"),
            GrammarChange::AddRule("YardstickAtWhere <- 'WHERE' Expression"),
            GrammarChange::PrependChoice("Indirection", "YardstickAtModifier"),
            // A column-name alias cannot consume FROM in an ordinary AS measure
            // projection. Reserved labels can still use the legacy frontend.
            GrammarChange::AddRule("YardstickMeasureAlias <- Expression 'AS' 'MEASURE' ColIdOrString", StartMeasure),
            GrammarChange::PrependChoice("AliasedExpression", "YardstickMeasureAlias"),
        };
    }
};

struct YardstickGrammarInfo final : ParserExtensionInfo {
    shared_ptr<CompiledGrammar> grammar;
    shared_ptr<CompiledGrammar> base_grammar;
};

shared_ptr<CompiledGrammar> SelectGrammar(ParserExtensionInfo *info, const ParserOptions &options) {
    if (options.compiled_grammar && options.compiled_grammar->GetRule("YardstickAtModifier")) {
        return options.compiled_grammar;
    }
    if (!info) {
        return nullptr;
    }
    auto &yardstick = info->Cast<YardstickGrammarInfo>();
    // Only replace the database's default grammar. A different active grammar
    // or dialect that lacks Yardstick rules must retain its own parser.
    if (options.compiled_grammar && options.compiled_grammar != yardstick.base_grammar) {
        return nullptr;
    }
    return yardstick.grammar;
}

} // namespace

shared_ptr<ParserExtensionInfo> RegisterYardstickGrammar(DatabaseInstance &db) {
    GrammarExtension::Register(db, make_shared_ptr<YardstickGrammar>());
    auto info = make_shared_ptr<YardstickGrammarInfo>();
    info->base_grammar = db.GetParserCache().GetMatcher();
    ClientContext context(db.shared_from_this());
    // Compile without changing any connection's active_grammar_extensions.
    info->grammar = CompiledGrammar::Create(context, {"yardstick"});
    return info;
}

NativeYardstickParseScope::NativeYardstickParseScope(ParserExtensionInfo *info, const ParserOptions &options_p)
    : options(options_p), previous(active_parse_scope), available(false) {
    auto grammar = SelectGrammar(info, options);
    if (grammar) {
        options.compiled_grammar = std::move(grammar);
        available = true;
    }
    // Parser callbacks are the caller of this adapter, not part of its inner
    // parses. Clearing them prevents re-entering semantic lowering recursively.
    options.extensions = nullptr;
    active_parse_scope = this;
}

NativeYardstickParseScope::~NativeYardstickParseScope() {
    active_parse_scope = previous;
}

const ParserOptions *CurrentNativeYardstickParserOptions() {
    return active_parse_scope ? &active_parse_scope->ParserConfig() : nullptr;
}

NativeYardstickBindScope::NativeYardstickBindScope(ClientContext &context) : previous(active_bind_context) {
    active_bind_context = &context;
}

NativeYardstickBindScope::~NativeYardstickBindScope() {
    active_bind_context = previous;
}

ClientContext *CurrentNativeYardstickClientContext() {
    return active_bind_context;
}

namespace {
thread_local const vector<unique_ptr<QueryNode>> *active_binding_ctes = nullptr;
}

const vector<unique_ptr<QueryNode>> *CurrentNativeYardstickCteBindings() {
    return active_binding_ctes;
}

NativeYardstickCteBindScope::NativeYardstickCteBindScope(const vector<unique_ptr<QueryNode>> *context)
    : previous(active_binding_ctes) {
    active_binding_ctes = context;
}

NativeYardstickCteBindScope::NativeYardstickCteBindScope(const vector<string> &queries,
                                                     const ParserOptions &options)
    : previous(active_binding_ctes) {
    for (auto &sql : queries) {
        Parser parser(options);
        parser.ParseQuery(sql);
        if (parser.statements.size() != 1 || parser.statements[0]->type != StatementType::SELECT_STATEMENT) {
            throw ParserException("Expected a SELECT carrying native CTE definitions");
        }
        definitions.push_back(parser.statements[0]->Cast<SelectStatement>().node->Copy());
    }
    active_binding_ctes = &definitions;
}

NativeYardstickCteBindScope::~NativeYardstickCteBindScope() {
    active_binding_ctes = previous;
}

BoundStatement BindNativeYardstickProbe(QueryNode &probe, const vector<QueryNode *> &local_scopes) {
    auto context = CurrentNativeYardstickClientContext();
    if (!context) {
        throw BinderException("Native query binding requires the originating bind context");
    }
    // Mirror DuckDB's lazy CTE binder chain. Each definition retains its own
    // lexical parent, so an inner name cannot change an earlier CTE's meaning.
    // These plans are used only for schema inspection and are never executed.
    vector<unique_ptr<CommonTableExpressionInfo>> definitions;
    vector<shared_ptr<Binder>> binders {Binder::CreateBinder(*context)};
    auto add_scope = [&](QueryNode &scope) {
        for (auto &entry : scope.cte_map.map) {
            // Binding consumes parts of the AST; every probe needs its own copy.
            definitions.push_back(entry.second->Copy());
            auto &definition = *definitions.back();
            auto &parent = *binders.back();
            auto state = make_shared_ptr<CTEBindState>(parent, *definition.query_node, definition.aliases);
            auto child = Binder::CreateBinder(*context, parent);
            child->bind_context.AddCTEBinding(
                make_uniq<CTEBinding>(BindingAlias(entry.first), state, parent.GenerateTableIndex()));
            binders.push_back(std::move(child));
        }
    };
    if (active_binding_ctes) {
        for (auto &scope : *active_binding_ctes) {
            add_scope(*scope);
        }
    }
    for (auto *scope : local_scopes) {
        add_scope(*scope);
    }
    return binders.back()->Bind(probe);
}

bool ParseNativeYardstickQuery(const string &sql, Parser &parser) {
    if (!active_parse_scope || !active_parse_scope->available || Parser::NormalizeSQLString(sql) != sql) {
        return false;
    }
    SyntaxCapture capture;
    capture.source = &sql;
    CaptureScope capture_scope(capture);
    parser.ParseQuery(sql);
    return true;
}

YardstickCreateViewInfo *FindNativeYardstickMeasures(const char *sql_p) {
    if (!sql_p || !active_parse_scope || !active_parse_scope->available) {
        return nullptr;
    }
    try {
        string sql(sql_p);
        if (sql.size() > std::numeric_limits<uint32_t>::max() || Parser::NormalizeSQLString(sql) != sql) {
            return nullptr; // Byte offsets must refer to the exact input.
        }
        SyntaxCapture capture;
        capture.source = &sql;
        CaptureScope capture_scope(capture);
        Parser parser(active_parse_scope->ParserConfig());
        parser.ParseQuery(sql);
        // The extension splits statements before semantic lowering. A direct
        // multi-statement FFI call retains the legacy contract.
        if (parser.statements.size() != 1) {
            return nullptr;
        }
        auto &statement = *parser.statements[0];
        if (statement.type != StatementType::CREATE_STATEMENT) {
            return nullptr;
        }
        auto &create = statement.Cast<CreateStatement>();
        if (!create.info || create.info->type != CatalogType::VIEW_ENTRY) {
            return nullptr;
        }
        auto &view = create.info->Cast<CreateViewInfo>();
        std::sort(capture.measures.begin(), capture.measures.end(), [](const SyntaxCapture::Measure &left,
                                                                    const SyntaxCapture::Measure &right) {
            return left.start < right.start;
        });
        auto duplicate = [](const string &value) {
            auto *copy = static_cast<char *>(std::malloc(value.size() + 1));
            if (!copy) {
                throw std::bad_alloc();
            }
            std::memcpy(copy, value.c_str(), value.size() + 1);
            return copy;
        };
        std::unique_ptr<YardstickCreateViewInfo, decltype(&yardstick_free_create_view_info)> result(
            new YardstickCreateViewInfo {}, yardstick_free_create_view_info);
        result->clean_sql = duplicate(sql);
        // The shared catalog resolves the final unquoted identifier.
        result->view_name = duplicate(view.GetViewName().GetIdentifierName());
        result->native_parsed = true;
        result->is_measure_view = !capture.measures.empty();
        for (auto &measure : capture.measures) {
            bool top_level = false;
            if (view.query && view.query->node && view.query->node->type == QueryNodeType::SELECT_NODE) {
                auto &select = view.query->node->Cast<SelectNode>();
                for (auto &projection : select.select_list) {
                    top_level |= projection.get() == measure.marker;
                }
            }
            if (!top_level) {
                result->error = duplicate("AS MEASURE declarations must be in the top-level CREATE VIEW projection");
                return result.release();
            }
        }
        if (!capture.measures.empty() && !view.aliases.empty()) {
            // Publish header-renamed metadata only while binding, so a failed
            // replacement restores the definition that DuckDB still owns.
            result->requires_binding = true;
            auto &select = view.query->node->Cast<SelectNode>();
            std::function<bool(const ParsedExpression &)> contains_star = [&](const ParsedExpression &expression) {
                if (expression.GetExpressionClass() == ExpressionClass::SUBQUERY ||
                    expression.GetExpressionClass() == ExpressionClass::WINDOW) {
                    return false;
                }
                if (expression.GetExpressionClass() == ExpressionClass::FUNCTION &&
                    IsYardstickStandardAggregate(expression.Cast<FunctionExpression>().FunctionName().GetIdentifierName())) {
                    return false;
                }
                bool found = expression.GetExpressionClass() == ExpressionClass::STAR;
                ParsedExpressionIterator::EnumerateChildren(expression, [&](const ParsedExpression &child) {
                    found |= contains_star(child);
                });
                return found;
            };
            bool has_star = false;
            for (auto &projection : select.select_list) {
                has_star |= contains_star(*projection);
            }
            if (has_star && !active_bind_context) {
                result->error = duplicate("AS MEASURE column lists with star expansion require the originating bind context");
                return result.release();
            }
            if (has_star) {
                // Bind only a layout probe in the originating transaction.
                // Placeholders retain the positions of ordinary projections;
                // DuckDB expands stars against the actual FROM/CTE bindings.
                auto probe = select.Copy();
                auto &probe_select = probe->Cast<SelectNode>();
                probe_select.groups = GroupByNode();
                probe_select.having.reset();
                probe_select.qualify.reset();
                probe_select.where_clause.reset();
                probe_select.modifiers.clear();
                probe_select.aggregate_handling = AggregateHandling::STANDARD_HANDLING;
                string prefix = "__yardstick_projection_";
                while (sql.find(prefix) != string::npos) {
                    prefix += "_";
                }
                vector<string> placeholders(select.select_list.size());
                for (idx_t i = 0; i < select.select_list.size(); i++) {
                    if (contains_star(*select.select_list[i])) {
                        continue;
                    }
                    placeholders[i] = prefix + std::to_string(i);
                    auto placeholder = ConstantExpression::FromValue(Value());
                    placeholder->SetAlias(Identifier(placeholders[i]));
                    probe_select.select_list[i] = std::move(placeholder);
                }
                try {
                    auto binder = Binder::CreateBinder(*active_bind_context);
                    auto bound = binder->Bind(*probe);
                    vector<unique_ptr<ParsedExpression>> expanded;
                    vector<bool> restored_placeholders(placeholders.size(), false);
                    for (auto &projection : bound.extra_info.original_expressions) {
                        bool restored = false;
                        if (projection->GetExpressionClass() == ExpressionClass::CONSTANT && projection->HasAlias()) {
                            auto alias = projection->GetAlias().GetIdentifierName();
                            for (idx_t i = 0; i < placeholders.size(); i++) {
                                if (!placeholders[i].empty() && placeholders[i] == alias) {
                                    if (restored_placeholders[i]) {
                                        throw ParserException("Ambiguous expanded AS MEASURE view projection");
                                    }
                                    expanded.push_back(std::move(select.select_list[i]));
                                    restored_placeholders[i] = true;
                                    restored = true;
                                    break;
                                }
                            }
                        }
                        if (!restored) {
                            expanded.push_back(std::move(projection));
                        }
                    }
                    if (expanded.size() != bound.names.size()) {
                        throw ParserException("Unable to map expanded AS MEASURE view columns");
                    }
                    for (idx_t i = 0; i < placeholders.size(); i++) {
                        if (!placeholders[i].empty() && !restored_placeholders[i]) {
                            throw ParserException("Missing expanded AS MEASURE view projection");
                        }
                    }
                    select.select_list = std::move(expanded);
                } catch (const std::exception &error) {
                    result->error = duplicate(error.what());
                    return result.release();
                }
            }
            if (view.aliases.size() > select.select_list.size()) {
                result->error = duplicate("More VIEW aliases than columns in query result");
                return result.release();
            }
            // Derived expressions retain their original declaration namespace
            // even when the header renames exposed columns. Expand references
            // before renaming, with aggregate argument and subquery boundaries.
            for (idx_t i = 0; i < capture.measures.size(); i++) {
                auto &declaration = capture.measures[i];
                auto expression = declaration.marker->Cast<FunctionExpression>()
                    .GetArguments()[0].GetExpression().Copy();
                vector<idx_t> active {i};
                try {
                    if (ExpandDeclarationReferences(expression, capture.measures, active)) {
                        declaration.expression = expression->ToString();
                    }
                } catch (const ParserException &error) {
                    result->error = duplicate(error.what());
                    return result.release();
                }
            }
            auto metadata_statement = view.query->Copy();
            auto &metadata = metadata_statement->Cast<SelectStatement>().node->Cast<SelectNode>();
            for (idx_t i = 0; i < select.select_list.size(); i++) {
                auto &projection = select.select_list[i];
                auto declaration = std::find_if(capture.measures.begin(), capture.measures.end(),
                                                [&](const SyntaxCapture::Measure &measure) {
                    return measure.marker == projection.get();
                });
                if (declaration != capture.measures.end()) {
                    metadata.select_list[i] = projection->Cast<FunctionExpression>().GetArguments()[0].GetExpression().Copy();
                    metadata.select_list[i]->SetAlias(projection->GetAlias());
                }
                if (i >= view.aliases.size()) {
                    continue;
                }
                auto &alias = view.aliases[i];
                // Only metadata receives the new SELECT aliases. Executable SQL
                // keeps the header and original aliases used by GROUP BY/HAVING.
                metadata.select_list[i]->SetAlias(alias);
                if (declaration != capture.measures.end()) {
                    declaration->name = alias.GetIdentifierName();
                }
                for (auto &group : metadata.groups.group_expressions) {
                    auto group_sql = group->ToString();
                    bool matches_alias = projection->HasAlias() &&
                        group->GetExpressionClass() == ExpressionClass::COLUMN_REF &&
                        group->Cast<ColumnRefExpression>().ColumnNames().size() == 1 &&
                        group->Cast<ColumnRefExpression>().ColumnNames()[0] == projection->GetAlias();
                    auto original = projection->Copy();
                    original->ClearAlias();
                    if (matches_alias || group_sql == original->ToString()) {
                        group = make_uniq<ColumnRefExpression>(alias);
                    }
                }
            }
            result->metadata_query_sql = duplicate(metadata_statement->ToString());
        }
        if (!capture.measures.empty()) {
            result->measures = new YardstickMeasureDef[capture.measures.size()] {};
            result->measure_count = capture.measures.size();
        }
        for (idx_t i = 0; i < capture.measures.size(); i++) {
            auto &source = capture.measures[i];
            auto &measure = result->measures[i];
            measure.column_name = duplicate(source.name);
            measure.alias_sql = duplicate(source.alias);
            measure.expression = duplicate(source.expression);
            measure.expr_start = static_cast<uint32_t>(source.start);
            measure.name_end = static_cast<uint32_t>(source.end);
        }
        return result.release();
    } catch (const std::exception &) {
        return nullptr;
    }
}

namespace {
vector<reference<ParseResult>> QuerySyntaxChildren(ParseResult &result) {
    switch (result.type) {
    case ParseResultType::LIST:
        return result.Cast<ListParseResult>().GetChildren();
    case ParseResultType::REPEAT:
        return result.Cast<RepeatParseResult>().GetChildren();
    case ParseResultType::CHOICE:
        return {result.Cast<ChoiceParseResult>().GetResult()};
    case ParseResultType::OPTIONAL:
        if (result.Cast<OptionalParseResult>().HasResult()) {
            return {result.Cast<OptionalParseResult>().GetResult()};
        }
        return {};
    default:
        return {};
    }
}

struct NativeQueryScope {
    idx_t start;
    idx_t end;
    vector<string> visible_ctes;
    vector<YardstickCteDefinition> cte_definitions;
};

// QueryNode does not retain a complete source location. The native parse tree
// owns query boundaries, while subsequent AST parsing owns expression semantics.
// Walk the tree without changing core grammar rules or rendering caller SQL.
bool CollectNativeQueryScopes(ParseResult &root, PEGTransformer &transformer,
                              vector<NativeQueryScope> &scopes) {
    struct Work {
        ParseResult *node;
        vector<string> visible_ctes;
        vector<YardstickCteDefinition> cte_definitions;
        QueryLocation main_query;
        idx_t modifier_end = 0;
    };
    vector<Work> pending {{&root, {}, {}, {}, 0}};
    while (!pending.empty()) {
        auto work = std::move(pending.back());
        pending.pop_back();
        auto &node = *work.node;
        if (node.name == "YardstickMeasureAlias") {
            return false; // Declaration registration owns this statement.
        }
        auto children = QuerySyntaxChildren(node);
        if (node.type == ParseResultType::LIST && node.name == "SimpleSelect") {
            auto location = node.GetLocation();
            if (!location.IsValid() || location.Start() == location.End()) {
                return false;
            }
            idx_t end = location.End();
            if (work.main_query.IsValid() && location.Start() == work.main_query.Start() &&
                end == work.main_query.End()) {
                end = MaxValue(end, work.modifier_end);
            }
            scopes.push_back({location.Start(), end, work.visible_ctes, work.cte_definitions});
        }
        // WITH is the first optional child of SELECT/INSERT/UPDATE/DELETE.
        // Handle its declarations in order: a non-recursive body sees earlier
        // CTEs, while a recursive body also sees its own name.
        idx_t first_child = 0;
        if (node.type == ParseResultType::LIST && !children.empty() &&
            children[0].get().type == ParseResultType::OPTIONAL) {
            auto &optional_with = children[0].get().Cast<OptionalParseResult>();
            if (optional_with.HasResult() && optional_with.GetResult().name == "WithClause") {
                auto &with = optional_with.GetResult().Cast<ListParseResult>();
                bool recursive = with.Child<OptionalParseResult>(1).HasResult();
                vector<ParseResult *> declarations;
                vector<ParseResult *> search {&with.GetChild(2)};
                while (!search.empty()) {
                    auto *candidate = search.back();
                    search.pop_back();
                    if (candidate->type == ParseResultType::LIST && candidate->name == "WithStatement") {
                        declarations.push_back(candidate);
                        continue;
                    }
                    auto descendants = QuerySyntaxChildren(*candidate);
                    for (auto child = descendants.rbegin(); child != descendants.rend(); ++child) {
                        search.push_back(&child->get());
                    }
                }
                for (auto *declaration : declarations) {
                    auto &list = declaration->Cast<ListParseResult>();
                    auto name = transformer.Transform<Identifier>(list.GetChild(0)).GetIdentifierName();
                    auto visible = work.visible_ctes;
                    auto definitions = work.cte_definitions;
                    auto location = declaration->GetLocation();
                    YardstickCteDefinition definition {static_cast<uint32_t>(location.Start()),
                                                      static_cast<uint32_t>(location.End()), recursive};
                    if (recursive) {
                        visible.push_back(name);
                        definitions.push_back(definition);
                    }
                    // Visiting the declaration also finds CTE bodies nested in
                    // DML; it never infers a query from parentheses or text.
                    pending.push_back({declaration, std::move(visible), std::move(definitions), {}, 0});
                    work.visible_ctes.push_back(std::move(name));
                    work.cte_definitions.push_back(definition);
                }
                first_child = 1;
            }
        }
        if (node.type == ParseResultType::LIST && node.name == "SelectStatementInternal") {
            work.main_query = node.Cast<ListParseResult>().GetChild(1).GetLocation();
            work.modifier_end = node.GetLocation().End();
        }
        for (idx_t i = first_child; i < children.size(); i++) {
            pending.push_back({&children[i].get(), work.visible_ctes, work.cte_definitions,
                               work.main_query, work.modifier_end});
        }
    }
    return true;
}
} // namespace

YardstickQueryScopeList *FindNativeYardstickQueryScopes(const char *sql_p) {
    if (!sql_p || !active_parse_scope || !active_parse_scope->available) {
        return nullptr;
    }
    try {
        string sql(sql_p);
        if (sql.size() > std::numeric_limits<uint32_t>::max() || Parser::NormalizeSQLString(sql) != sql) {
            return nullptr;
        }
        auto options = active_parse_scope->ParserConfig();
        auto &grammar = *options.compiled_grammar;
        vector<MatcherToken> tokens;
        TokenizerBehavior behavior(sql, tokens);
        grammar.GetTokenizer().TokenizeInput(behavior);
        TokenIterator iterator(tokens);
        vector<NativeQueryScope> scopes;
        while (iterator.HasMoreStatements()) {
            vector<MatcherSuggestion> suggestions;
            ParseResultAllocator results;
            ParserPackratCache packrat;
            idx_t max_token = iterator.Position();
            ArenaAllocator process_allocator(Allocator::DefaultAllocator());
            MatchContext context(suggestions, results, process_allocator, max_token, MatchMode::BUILD_PARSE_RESULT,
                                 options.identifier_case_mode, options.heap_based_parser, &packrat);
            MatchState state(iterator, context);
            auto match = grammar.TopLevelStatementMatcher().MatchParseResult(state);
            if (!match.IsSuccess() || !match.HasParseResult() || state.token_iterator.Position() <= iterator.Position()) {
                return nullptr;
            }
            iterator.SetPosition(state.token_iterator);
            ArenaAllocator transform_allocator(Allocator::DefaultAllocator());
            PEGTransformer transformer(transform_allocator, iterator, options, grammar);
            if (!CollectNativeQueryScopes(*match.GetParseResult(), transformer, scopes)) {
                return nullptr;
            }
        }
        std::sort(scopes.begin(), scopes.end(), [](const NativeQueryScope &left, const NativeQueryScope &right) {
            return left.start != right.start ? left.start < right.start : left.end > right.end;
        });
        scopes.erase(std::unique(scopes.begin(), scopes.end(), [](const NativeQueryScope &left,
                                                                const NativeQueryScope &right) {
            return left.start == right.start && left.end == right.end;
        }), scopes.end());
        std::unique_ptr<YardstickQueryScopeList, decltype(&yardstick_free_query_scopes)> output(
            new YardstickQueryScopeList {}, yardstick_free_query_scopes);
        if (!scopes.empty()) {
            output->scopes = new YardstickQueryScope[scopes.size()] {};
            output->count = scopes.size();
        }
        for (idx_t i = 0; i < scopes.size(); i++) {
            auto &source = scopes[i];
            auto &scope = output->scopes[i];
            scope.start_pos = static_cast<uint32_t>(source.start);
            scope.end_pos = static_cast<uint32_t>(source.end);
            if (!source.visible_ctes.empty()) {
                scope.visible_ctes = new const char *[source.visible_ctes.size()] {};
                scope.cte_definitions = new YardstickCteDefinition[source.cte_definitions.size()] {};
                scope.visible_cte_count = source.visible_ctes.size();
                for (idx_t j = 0; j < source.visible_ctes.size(); j++) {
                    scope.visible_ctes[j] = strdup(source.visible_ctes[j].c_str());
                    scope.cte_definitions[j] = source.cte_definitions[j];
                    if (!scope.visible_ctes[j]) {
                        throw std::bad_alloc();
                    }
                }
            }
        }
        return output.release();
    } catch (const std::exception &) {
        return nullptr;
    }
}

YardstickCurrentReferenceList *FindNativeYardstickCurrentReferences(const char *expression) {
    if (!expression || !active_parse_scope || !active_parse_scope->available) {
        return nullptr;
    }
    const string prefix = "SELECT AGGREGATE(__yardstick_probe) AT (SET __yardstick_dimension = ";
    string sql = prefix + expression + ")";
    if (sql.size() > std::numeric_limits<uint32_t>::max() || Parser::NormalizeSQLString(sql) != sql) {
        return nullptr;
    }
    std::unique_ptr<YardstickCurrentReferenceList, decltype(&yardstick_free_current_reference_list)> output(
        new YardstickCurrentReferenceList {}, yardstick_free_current_reference_list);
    try {
        SyntaxCapture capture;
        capture.source = &sql;
        CaptureScope capture_scope(capture);
        Parser parser(active_parse_scope->ParserConfig());
        parser.ParseQuery(sql);
        auto &references = capture.current_references;
        std::sort(references.begin(), references.end(), [](const SyntaxCapture::CurrentReference &left,
                                                         const SyntaxCapture::CurrentReference &right) {
            return left.start < right.start;
        });
        if (!references.empty()) {
            output->references = new YardstickCurrentReference[references.size()] {};
            output->count = references.size();
        }
        for (idx_t i = 0; i < references.size(); i++) {
            auto &source = references[i];
            if (source.start < prefix.size() || source.end > sql.size() - 1) {
                throw ParserException("CURRENT reference is outside its modifier expression");
            }
            auto *dimension = static_cast<char *>(std::malloc(source.dimension.size() + 1));
            if (!dimension) {
                throw std::bad_alloc();
            }
            std::memcpy(dimension, source.dimension.c_str(), source.dimension.size() + 1);
            output->references[i] = {dimension, static_cast<uint32_t>(source.start - prefix.size()),
                                    static_cast<uint32_t>(source.end - prefix.size())};
        }
    } catch (const std::exception &error) {
        output->error = strdup(error.what());
    }
    return output.release();
}

YardstickAggregateCallList *FindNativeYardstickAggregates(const char *sql_p) {
    if (!sql_p || !active_parse_scope || !active_parse_scope->available) {
        return nullptr;
    }
    bool found_decorated_call = false;
    try {
        string sql(sql_p);
        if (sql.size() > std::numeric_limits<uint32_t>::max() || Parser::NormalizeSQLString(sql) != sql) {
            // The C ABI reports offsets into the exact caller-owned byte string.
            return nullptr;
        }
        SyntaxCapture capture;
        capture.source = &sql;
        CaptureScope capture_scope(capture);
        Parser parser(active_parse_scope->ParserConfig());
        parser.ParseQuery(sql);
        if (capture.has_measure) {
            return nullptr;
        }

        struct Aggregate {
            string measure;
            idx_t start;
            idx_t end;
            vector<NativeModifier> modifiers;
            string call_sql;
            bool is_window;
            bool has_decorations;
        };
        vector<Aggregate> aggregates;
        vector<bool> used_clauses(capture.clauses.size(), false);
        auto source_range = [&](const ParsedExpression &expression) {
            auto location = expression.GetQueryLocation();
            if (!location.IsValid() || !location.length || location.offset > sql.size() ||
                location.length > sql.size() - location.offset) {
                throw ParserException("Yardstick aggregate source location is unavailable");
            }
            return location;
        };
        vector<MatcherToken> source_tokens;
        TokenizerBehavior token_behavior(sql, source_tokens);
        active_parse_scope->ParserConfig().compiled_grammar->GetTokenizer().TokenizeInput(token_behavior);
        source_tokens.erase(std::remove_if(source_tokens.begin(), source_tokens.end(), [](const MatcherToken &token) {
            return token.type == TokenType::COMMENT || token.type == TokenType::END_OF_INPUT;
        }), source_tokens.end());
        auto extend_operand = [&](idx_t &start, idx_t end, idx_t suffix_start) {
            // The AST erases grouping parentheses. Consume a closing parenthesis
            // before AT only with its immediately enclosing opening parenthesis.
            // Calls and suffix parentage still come exclusively from the AST.
            auto first = std::lower_bound(source_tokens.begin(), source_tokens.end(), start,
                [](const MatcherToken &token, idx_t offset) { return token.offset < offset; });
            auto next = std::lower_bound(source_tokens.begin(), source_tokens.end(), end,
                [](const MatcherToken &token, idx_t offset) { return token.offset < offset; });
            while (next != source_tokens.end() && next->offset < suffix_start) {
                if (next->text != ")" || first == source_tokens.begin() || (first - 1)->text != "(") {
                    return false;
                }
                start = (--first)->offset;
                ++next;
            }
            return next != source_tokens.end() && next->offset == suffix_start;
        };
        std::function<void(unique_ptr<ParsedExpression> &)> visit_expression;
        visit_expression = [&](unique_ptr<ParsedExpression> &expression) {
            auto *base = expression.get();
            vector<idx_t> suffixes;
            while (true) {
                auto clause = std::find_if(capture.clauses.begin(), capture.clauses.end(),
                                          [&](const NativeAtClause &item) { return item.marker == base; });
                if (clause == capture.clauses.end()) {
                    break;
                }
                auto index = static_cast<idx_t>(clause - capture.clauses.begin());
                if (used_clauses[index] || base->GetExpressionClass() != ExpressionClass::FUNCTION) {
                    throw ParserException("Ambiguous Yardstick AT expression");
                }
                auto &arguments = base->Cast<FunctionExpression>().GetArgumentsMutable();
                if (arguments.size() != 1) {
                    throw ParserException("Unsupported Yardstick AT expression");
                }
                used_clauses[index] = true;
                suffixes.push_back(index);
                base = arguments[0].GetExpressionMutable().get();
            }

            const vector<FunctionArgument> *arguments = nullptr;
            bool is_window = false;
            bool has_decorations = false;
            if (base->GetExpressionClass() == ExpressionClass::FUNCTION) {
                auto &function = base->Cast<FunctionExpression>();
                if (StringUtil::CIEquals(function.FunctionName().GetIdentifierName(), "aggregate")) {
                    arguments = &function.GetArguments();
                    has_decorations = function.Distinct() || function.Filter() || function.ExportState() ||
                                      (function.OrderBy() && !function.OrderBy()->orders.empty());
                }
            } else if (base->GetExpressionClass() == ExpressionClass::WINDOW) {
                auto &window = base->Cast<WindowExpression>();
                if (StringUtil::CIEquals(window.FunctionName().GetIdentifierName(), "aggregate")) {
                    arguments = &window.GetArguments();
                    is_window = true;
                    has_decorations = true;
                }
            }
            bool is_measure_call = arguments && arguments->size() == 1;
            if (is_measure_call) {
                found_decorated_call |= has_decorations;
                auto call_location = source_range(*base);
                auto &argument = (*arguments)[0];
                auto argument_location = source_range(argument.GetExpression());
                if (argument.HasName() || argument_location.Start() < call_location.Start() ||
                    argument_location.End() > call_location.End()) {
                    throw ParserException("Unsupported Yardstick aggregate argument source");
                }
                Aggregate aggregate {sql.substr(argument_location.Start(), argument_location.length),
                                     call_location.Start(), call_location.End(), {}, base->ToString(),
                                     is_window, has_decorations};
                // AST parents run from the last suffix back to the first.
                // Modifier application retains the original SQL order.
                for (auto suffix = suffixes.rbegin(); suffix != suffixes.rend(); ++suffix) {
                    auto &clause = capture.clauses[*suffix];
                    if (clause.start < aggregate.end || clause.end > sql.size() ||
                        !extend_operand(aggregate.start, aggregate.end, clause.start)) {
                        throw ParserException("Invalid Yardstick AT source range");
                    }
                    aggregate.modifiers.insert(aggregate.modifiers.end(), clause.modifiers.begin(),
                                               clause.modifiers.end());
                    aggregate.end = clause.end;
                }
                aggregates.push_back(std::move(aggregate));
            }
            if (!suffixes.empty() && !is_measure_call) {
                // Shorthand AT expressions still use compatibility lowering.
                throw ParserException("Yardstick AT operand is not an aggregate call");
            }
            if (base->GetExpressionClass() == ExpressionClass::SUBQUERY) {
                auto &subquery = base->Cast<SubqueryExpression>();
                ParsedExpressionIterator::EnumerateQueryNodeChildren(*subquery.SubqueryMutable()->node,
                                                                     visit_expression);
            }
            // Includes arguments of DuckDB's multiargument aggregate(list, name)
            // so nested measure calls remain visible without claiming that call.
            ParsedExpressionIterator::EnumerateChildren(*base, visit_expression);
        };
        for (auto &statement : parser.statements) {
            if (!EnumerateNativeStatementExpressions(*statement, visit_expression)) {
                return nullptr;
            }
        }
        std::sort(aggregates.begin(), aggregates.end(), [](const Aggregate &left, const Aggregate &right) {
            return left.start < right.start;
        });
        for (idx_t i = 1; i < aggregates.size(); i++) {
            if (aggregates[i].start < aggregates[i - 1].end) {
                // Overlapping measure replacements require another lowering step.
                return nullptr;
            }
        }
        if (std::find(used_clauses.begin(), used_clauses.end(), false) != used_clauses.end()) {
            // Captured modifier expressions may contain syntax outside the AST
            // marker operand. Never silently omit those references.
            return nullptr;
        }

        auto duplicate = [](const string &value) {
            auto *copy = static_cast<char *>(std::malloc(value.size() + 1));
            if (!copy) {
                throw std::bad_alloc();
            }
            std::memcpy(copy, value.c_str(), value.size() + 1);
            return copy;
        };
        std::unique_ptr<YardstickAggregateCallList, decltype(&yardstick_free_aggregate_list)> result(
            new YardstickAggregateCallList {}, yardstick_free_aggregate_list);
        result->native_parsed = true;
        if (!aggregates.empty()) {
            result->calls = new YardstickAggregateCall[aggregates.size()] {};
            result->count = aggregates.size();
        }
        for (idx_t i = 0; i < aggregates.size(); i++) {
            auto &source = aggregates[i];
            auto &call = result->calls[i];
            call.measure_name = duplicate(source.measure);
            call.call_sql = duplicate(source.call_sql);
            call.is_window = source.is_window;
            call.has_decorations = source.has_decorations;
            call.start_pos = static_cast<uint32_t>(source.start);
            call.end_pos = static_cast<uint32_t>(source.end);
            if (!source.modifiers.empty()) {
                call.modifiers = new YardstickAtModifier[source.modifiers.size()] {};
                call.modifier_count = source.modifiers.size();
            }
            for (idx_t j = 0; j < source.modifiers.size(); j++) {
                auto &modifier = source.modifiers[j];
                auto &output = call.modifiers[j];
                output.type = modifier.type;
                if (modifier.type == YARDSTICK_AT_ALL_DIM || modifier.type == YARDSTICK_AT_SET) {
                    output.dimension = duplicate(modifier.dimension);
                }
                if (modifier.type == YARDSTICK_AT_WHERE || modifier.type == YARDSTICK_AT_SET) {
                    output.value = duplicate(modifier.value);
                }
            }
        }
        return result.release();
    } catch (const std::exception &error) {
        if (found_decorated_call) {
            auto *result = new YardstickAggregateCallList {};
            result->native_parsed = true;
            result->error = strdup(error.what());
            return result;
        }
        return nullptr;
    }
}

bool NormalizeYardstickGrammar(ParserExtensionInfo *info, const ParserOptions &options,
                              string &sql, bool &has_measure) {
    auto grammar = SelectGrammar(info, options);
    if (!grammar) {
        return false;
    }
    vector<MatcherToken> tokens;
    TokenizerBehavior behavior(sql, tokens);
    grammar->GetTokenizer().TokenizeInput(behavior);
    bool has_custom_syntax = false;
    for (idx_t i = 1; i < tokens.size(); i++) {
        has_custom_syntax |=
            (StringUtil::CIEquals(tokens[i - 1].text, "AT") && tokens[i].text == "(") ||
            (StringUtil::CIEquals(tokens[i - 1].text, "AS") && StringUtil::CIEquals(tokens[i].text, "MEASURE"));
    }
    if (!has_custom_syntax) {
        has_measure = false;
        return true;
    }
    ParserOptions native_options = options;
    native_options.extensions = nullptr;
    native_options.compiled_grammar = grammar;
    SyntaxCapture capture;
    CaptureScope scope(capture);
    try {
        Parser parser(native_options);
        parser.ParseQuery(sql);
    } catch (const ParserException &) {
        // Retain forms outside this grammar during the frontend migration.
        return false;
    }
    has_measure = capture.has_measure;
    if (capture.spans.empty()) {
        return true;
    }

    // Include trivia between the preceding expression and its postfix/alias.
    // Otherwise a comment before AT would still hide it from the legacy bridge.
    for (auto &span : capture.spans) {
        idx_t previous_end = span.start;
        for (auto &token : tokens) {
            if (token.offset >= span.start) {
                break;
            }
            if (token.type != TokenType::COMMENT) {
                previous_end = token.offset + token.length;
            }
        }
        span.start = previous_end;
    }
    std::sort(capture.spans.begin(), capture.spans.end(), [](const SyntaxSpan &left, const SyntaxSpan &right) {
        return left.start < right.start;
    });
    // Replace only trivia inside native syntax spans. Token bytes (including
    // quotes, dollar strings, and UTF-8) and all surrounding SQL stay intact.
    string normalized;
    idx_t cursor = 0;
    for (auto &span : capture.spans) {
        if (span.start < cursor) {
            continue; // An enclosing AT span already includes this child.
        }
        normalized.append(sql, cursor, span.start - cursor);
        idx_t token_end = span.start;
        bool first_token = true;
        for (auto &token : tokens) {
            if (token.offset < span.start || token.offset >= span.end || token.type == TokenType::COMMENT ||
                token.type == TokenType::END_OF_INPUT) {
                continue;
            }
            if (first_token || span.measure || token.offset > token_end) {
                normalized += !span.measure && token.preceded_by_newline ? '\n' : ' ';
            }
            normalized.append(sql, token.offset, token.length);
            token_end = token.offset + token.length;
            first_token = false;
        }
        cursor = span.end;
    }
    normalized.append(sql, cursor, sql.size() - cursor);
    sql = std::move(normalized);
    return true;
}

} // namespace duckdb
#endif
