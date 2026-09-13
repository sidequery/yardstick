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

namespace duckdb {
namespace {

struct SyntaxSpan {
    idx_t start;
    idx_t end;
    bool measure;
};

struct SyntaxCapture {
    vector<SyntaxSpan> spans;
    bool has_measure = false;
};

// Grammar callbacks have no per-parse extension state. This scope contains only
// source locations, never catalog state, and restores nested parses on exit.
thread_local SyntaxCapture *active_capture = nullptr;

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

unique_ptr<TransformResultValue> TransformAt(PEGTransformer &, ParseResult &result) {
    CaptureSpan(result, result);
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
            GrammarChange::AddRule("YardstickMeasureAlias <- Expression 'AS' 'MEASURE' ColLabelOrString", StartMeasure),
            GrammarChange::PrependChoice("AliasedExpression", "YardstickMeasureAlias"),
        };
    }
};

struct YardstickGrammarInfo final : ParserExtensionInfo {
    shared_ptr<CompiledGrammar> grammar;
};

} // namespace

shared_ptr<ParserExtensionInfo> RegisterYardstickGrammar(DatabaseInstance &db) {
    GrammarExtension::Register(db, make_shared_ptr<YardstickGrammar>());
    auto info = make_shared_ptr<YardstickGrammarInfo>();
    ClientContext context(db.shared_from_this());
    // Compile without changing any connection's active_grammar_extensions.
    info->grammar = CompiledGrammar::Create(context, {"yardstick"});
    return info;
}

bool NormalizeYardstickGrammar(ParserExtensionInfo *info, const ParserOptions &options,
                              string &sql, bool &has_measure) {
    if (!info) {
        return false;
    }
    auto grammar = info->Cast<YardstickGrammarInfo>().grammar;
    if (options.compiled_grammar && options.compiled_grammar->HasGrammarChanges()) {
        // Preserve other activated extensions. If their combined grammar does
        // not include Yardstick, the existing frontend remains available.
        if (!options.compiled_grammar->GetRule("YardstickAtModifier")) {
            return false;
        }
        grammar = options.compiled_grammar;
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
        // Retain legacy dialect forms (including CURRENT dimension references
        // and brace shorthand) during this staged frontend migration.
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
