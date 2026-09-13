#include "frontend_peg.hpp"

#if YARDSTICK_GRAMMAR_EXTENSION
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/parser/grammar_extension.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/parser/peg/compiled_grammar.hpp"
#include "duckdb/parser/peg/transformer/peg_transformer.hpp"

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
};

struct SyntaxCapture {
    vector<SyntaxSpan> spans;
    bool has_measure = false;
    const string *source = nullptr;
    vector<NativeAtClause> clauses;
};

// Grammar callbacks have no per-parse extension state. This scope captures only
// source syntax, never catalog state, and restores nested parses on exit.
thread_local SyntaxCapture *active_capture = nullptr;
thread_local const NativeYardstickParseScope *active_parse_scope = nullptr;

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
    return transformer.Transform<unique_ptr<ParsedExpression>>(result)->ToString();
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
    if (active_capture->source) {
        NativeAtClause clause {result.offset.GetIndex(), result.offset.GetIndex() + result.length.GetIndex(), {}};
        auto &list = result.Cast<ListParseResult>();
        for (auto &child : list.Child<RepeatParseResult>(2).GetChildren()) {
            ReadModifier(transformer, child.get(), clause.modifiers);
        }
        active_capture->clauses.push_back(std::move(clause));
    }
    // BaseExpression inserts the left operand into this marker. The adapter
    // consumes source spans before binding; semantic lowering stays in Rust.
    unique_ptr<ParsedExpression> expression =
        make_uniq<FunctionExpression>("__yardstick_at", vector<unique_ptr<ParsedExpression>> {});
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
    }
    return make_uniq<TypedTransformResult<unique_ptr<ParsedExpression>>>(std::move(expression));
}

unique_ptr<TransformProcess> StartAt(PEGTransformer &transformer, ParseResult &result) {
    return make_uniq<FinalizeTransformProcess>(transformer, result, TransformAt);
}

unique_ptr<TransformProcess> StartMeasure(PEGTransformer &transformer, ParseResult &result) {
    return make_uniq<FinalizeTransformProcess>(transformer, result, TransformMeasure);
}

class YardstickGrammar final : public GrammarExtension {
public:
    YardstickGrammar() : GrammarExtension("yardstick", "Yardstick measure aliases and context modifiers") {}

    vector<GrammarChange> GetChanges() const override {
        return {
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
};

shared_ptr<CompiledGrammar> SelectGrammar(ParserExtensionInfo *info, const ParserOptions &options) {
    if (options.compiled_grammar && options.compiled_grammar->HasGrammarChanges()) {
        return options.compiled_grammar->GetRule("YardstickAtModifier") ? options.compiled_grammar : nullptr;
    }
    if (!info) {
        return nullptr;
    }
    return info->Cast<YardstickGrammarInfo>().grammar;
}

} // namespace

shared_ptr<ParserExtensionInfo> RegisterYardstickGrammar(DatabaseInstance &db) {
    GrammarExtension::Register(db, make_shared_ptr<YardstickGrammar>());
    auto info = make_shared_ptr<YardstickGrammarInfo>();
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

YardstickAggregateCallList *FindNativeYardstickAggregates(const char *sql_p) {
    if (!sql_p || !active_parse_scope || !active_parse_scope->available) {
        return nullptr;
    }
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

        vector<MatcherToken> tokens;
        TokenizerBehavior behavior(sql, tokens);
        active_parse_scope->ParserConfig().compiled_grammar->GetTokenizer().TokenizeInput(behavior);
        vector<const MatcherToken *> significant;
        for (auto &token : tokens) {
            if (token.type != TokenType::COMMENT && token.type != TokenType::END_OF_INPUT) {
                significant.push_back(&token);
            }
        }
        std::sort(capture.clauses.begin(), capture.clauses.end(), [](const NativeAtClause &left,
                                                                  const NativeAtClause &right) {
            return left.start < right.start;
        });
        vector<bool> used_clauses(capture.clauses.size(), false);
        struct Aggregate {
            string measure;
            idx_t start;
            idx_t end;
            vector<NativeModifier> modifiers;
        };
        vector<Aggregate> aggregates;
        for (idx_t i = 0; i + 1 < significant.size(); i++) {
            auto &name = *significant[i];
            if ((!StringUtil::CIEquals(name.text, "AGGREGATE") &&
                 !StringUtil::CIEquals(name.text, "\"AGGREGATE\"")) || significant[i + 1]->text != "(") {
                continue;
            }
            // The PEG parse establishes SQL validity. Token positions are used
            // only to associate function ranges with its already-parsed AT
            // suffixes; modifier contents come entirely from ParseResult nodes.
            idx_t depth = 1;
            idx_t close = i + 2;
            for (; close < significant.size(); close++) {
                auto &text = significant[close]->text;
                if (text == "(" || text == "[" || text == "{") {
                    depth++;
                } else if (text == ")" || text == "]" || text == "}") {
                    if (--depth == 0) {
                        break;
                    }
                } else if (text == "," && depth == 1) {
                    // DuckDB's list aggregate(list, function) is not a measure.
                    return nullptr;
                }
            }
            if (close >= significant.size() || close == i + 2) {
                return nullptr;
            }
            idx_t name_start = i;
            while (name_start >= 2 && significant[name_start - 1]->text == "." &&
                   (significant[name_start - 2]->type == TokenType::IDENTIFIER ||
                    significant[name_start - 2]->type == TokenType::KEYWORD)) {
                name_start -= 2;
            }
            auto argument_start = significant[i + 2]->offset;
            auto argument_end = significant[close - 1]->offset + significant[close - 1]->length;
            Aggregate aggregate {sql.substr(argument_start, argument_end - argument_start),
                                 significant[name_start]->offset,
                                 significant[close]->offset + significant[close]->length,
                                 {}};
            idx_t after = close + 1;
            while (after < significant.size() && StringUtil::CIEquals(significant[after]->text, "AT")) {
                auto clause = std::find_if(capture.clauses.begin(), capture.clauses.end(), [&](const NativeAtClause &item) {
                    return item.start == significant[after]->offset;
                });
                if (clause == capture.clauses.end()) {
                    return nullptr;
                }
                auto clause_index = static_cast<idx_t>(clause - capture.clauses.begin());
                if (used_clauses[clause_index]) {
                    return nullptr;
                }
                used_clauses[clause_index] = true;
                aggregate.modifiers.insert(aggregate.modifiers.end(), clause->modifiers.begin(), clause->modifiers.end());
                aggregate.end = clause->end;
                while (after < significant.size() && significant[after]->offset < clause->end) {
                    after++;
                }
            }
            if (!aggregates.empty() && aggregate.start < aggregates.back().end) {
                // Nested replacement ranges need a separate lowering step.
                return nullptr;
            }
            aggregates.push_back(std::move(aggregate));
        }
        if (std::find(used_clauses.begin(), used_clauses.end(), false) != used_clauses.end()) {
            // A suffix on a shorthand expression or nested inside another AT
            // clause must not be silently omitted from the structured result.
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
        if (!aggregates.empty()) {
            result->calls = new YardstickAggregateCall[aggregates.size()] {};
            result->count = aggregates.size();
        }
        for (idx_t i = 0; i < aggregates.size(); i++) {
            auto &source = aggregates[i];
            auto &call = result->calls[i];
            call.measure_name = duplicate(source.measure);
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
    } catch (const std::exception &) {
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
