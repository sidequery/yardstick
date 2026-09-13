#include "duckdb.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/parser/expression/cast_expression.hpp"
#include "duckdb/parser/grammar_extension.hpp"
#include "duckdb/parser/peg/transformer/peg_transformer.hpp"

using namespace duckdb;

namespace {

unique_ptr<TransformResultValue> TransformTestYear(PEGTransformer &, ParseResult &) {
    // Keep this an expression when the lowerer emits it in GROUP BY; a bare
    // integer literal would be interpreted as a select-list ordinal.
    unique_ptr<ParsedExpression> expression =
        make_uniq<CastExpression>(LogicalType::INTEGER, ConstantExpression::Integer(2023));
    return make_uniq<TypedTransformResult<unique_ptr<ParsedExpression>>>(std::move(expression));
}

unique_ptr<TransformProcess> StartTestYear(PEGTransformer &transformer, ParseResult &result) {
    return make_uniq<FinalizeTransformProcess>(transformer, result, TransformTestYear);
}

class TestGrammar final : public GrammarExtension {
public:
    TestGrammar() : GrammarExtension("yardstick_test_grammar", "Combined grammar regression fixture") {}

    vector<GrammarChange> GetChanges() const override {
        // The scalar expression survives the Rust SQL rewrite. The bang before
        // the parentheses makes it invalid without this grammar extension.
        return {
            GrammarChange::AddRule("YardstickTestYear <- 'YARDSTICK_TEST_YEAR' '!' '(' ')'", StartTestYear),
            GrammarChange::PrependChoice("SingleExpression", "YardstickTestYear"),
        };
    }
};

} // namespace

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(yardstick_test_grammar, loader) {
    GrammarExtension::Register(loader.GetDatabaseInstance(), make_shared_ptr<TestGrammar>());
}

}
