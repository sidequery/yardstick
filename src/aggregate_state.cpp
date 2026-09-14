#include "aggregate_state.hpp"

#if YARDSTICK_GRAMMAR_EXTENSION
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/function_binder.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/expression/columnref_expression.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/expression/bound_case_expression.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/expression_binder/constant_binder.hpp"

namespace duckdb {
namespace {

// Logical-type aliases are serialized with the type, including in persistent tables and prepared statements.
// The versioned alias owns the finalization expression; each sN field retains its native aggregate state type.
constexpr const char *STATE_ALIAS_PREFIX = "yardstick_state_v1:";

struct StateExpressionData : FunctionData {
	explicit StateExpressionData(unique_ptr<Expression> expression_p) : expression(std::move(expression_p)) {
	}
	unique_ptr<Expression> expression;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<StateExpressionData>(expression->Copy());
	}
	bool Equals(const FunctionData &other) const override {
		return expression->Equals(*other.Cast<StateExpressionData>().expression);
	}
};

unique_ptr<Expression> BindNative(ClientContext &context, const Identifier &name,
	                               vector<unique_ptr<Expression>> arguments) {
	FunctionBinder binder(context);
	ErrorData error;
	auto result = binder.BindScalarFunction(Identifier::DefaultSchema(), name, std::move(arguments), error);
	if (!result) {
		error.Throw();
	}
	return result;
}

unique_ptr<Expression> ExtractState(ClientContext &context, const Expression &state, const Identifier &name) {
	vector<unique_ptr<Expression>> arguments;
	// DuckDB does not implicitly cast an aliased STRUCT to struct_extract's STRUCT parameter.
	// Remove only the outer alias: native aggregate-state aliases on the fields must survive.
	auto struct_type = state.GetReturnType().WithAlias("");
	arguments.push_back(BoundCastExpression::AddCastToType(context, state.Copy(), struct_type));
	arguments.push_back(make_uniq<BoundConstantExpression>(Value(name.GetIdentifierName())));
	return BindNative(context, "struct_extract", std::move(arguments));
}

bool IsCompositeState(const LogicalType &type) {
	return type.id() == LogicalTypeId::STRUCT && type.HasAlias() &&
	       StringUtil::StartsWith(type.GetAlias(), STATE_ALIAS_PREFIX);
}

void ValidateFields(const LogicalType &type) {
	if (type.id() != LogicalTypeId::STRUCT || StructType::GetChildTypes(type).empty()) {
		throw BinderException("yardstick_state requires a nonempty STRUCT of native aggregate states");
	}
	auto &fields = StructType::GetChildTypes(type);
	for (idx_t i = 0; i < fields.size(); i++) {
		if (fields[i].first != "s" + std::to_string(i) || !fields[i].second.IsAggregateState()) {
			throw BinderException("yardstick_state fields must be native aggregate states named s0, s1, ... in order");
		}
	}
}

class StateFormulaBinder : public ConstantBinder {
public:
	StateFormulaBinder(Binder &binder, ClientContext &context, const Expression &state_p)
	    : ConstantBinder(binder, context, "yardstick_state formula"), state(state_p) {
	}

protected:
	BindResult BindExpression(unique_ptr<ParsedExpression> &expression, idx_t depth, bool root_expression) override {
		if (expression->GetExpressionClass() != ExpressionClass::COLUMN_REF) {
			return ConstantBinder::BindExpression(expression, depth, root_expression);
		}
		auto &reference = expression->Cast<ColumnRefExpression>();
		if (!reference.IsQualified()) {
			for (auto &field : StructType::GetChildTypes(state.GetReturnType())) {
				if (reference.GetColumnName() == field.first) {
					vector<unique_ptr<Expression>> arguments;
					arguments.push_back(ExtractState(context, state, field.first));
					return BindResult(BindNative(context, "finalize", std::move(arguments)));
				}
			}
		}
		throw BinderException("yardstick_state formula references unknown state field %s", reference.ToString());
	}

private:
	const Expression &state;
};

unique_ptr<ParsedExpression> ParseFormula(const string &formula) {
	auto expressions = Parser::ParseExpressionList(formula);
	if (expressions.size() != 1 || !expressions[0]->GetAlias().empty()) {
		throw BinderException("yardstick_state requires exactly one scalar finalization expression");
	}
	return std::move(expressions[0]);
}

unique_ptr<Expression> BindFormula(ClientContext &context, const Expression &state,
	                                unique_ptr<ParsedExpression> formula) {
	auto binder = Binder::CreateBinder(context);
	StateFormulaBinder expression_binder(*binder, context, state);
	return expression_binder.Bind(formula);
}

unique_ptr<Expression> IfNull(const Expression &input, unique_ptr<Expression> when_null,
	                           unique_ptr<Expression> otherwise) {
	auto condition = make_uniq<BoundOperatorExpression>(ExpressionType::OPERATOR_IS_NULL, LogicalType::BOOLEAN);
	condition->GetChildrenMutable().push_back(input.Copy());
	return make_uniq<BoundCaseExpression>(std::move(condition), std::move(when_null), std::move(otherwise));
}

unique_ptr<FunctionData> ReturnExpression(BindScalarFunctionInput &input, unique_ptr<Expression> expression) {
	input.GetBoundFunction().SetReturnType(expression->GetReturnType());
	return make_uniq<StateExpressionData>(std::move(expression));
}

unique_ptr<Expression> ReplaceStateExpression(FunctionBindExpressionInput &input) {
	return input.bind_data->Cast<StateExpressionData>().expression->Copy();
}

unique_ptr<FunctionData> BindState(BindScalarFunctionInput &input) {
	auto &state = *input.GetArguments()[1];
	ValidateFields(state.GetReturnType());
	auto formula = ParseFormula(input.GetConstant(0, false).ToString());
	auto canonical_formula = formula->ToString();
	// Validate the complete formula now, rather than storing a state that only fails at finalization time.
	BindFormula(input.GetClientContext(), state, std::move(formula));
	auto type = state.GetReturnType().WithAlias(string(STATE_ALIAS_PREFIX) + canonical_formula);
	auto result = BoundCastExpression::AddCastToType(input.GetClientContext(), state.Copy(), type);
	return ReturnExpression(input, std::move(result));
}

unique_ptr<FunctionData> BindFinalize(BindScalarFunctionInput &input) {
	auto &state = *input.GetArguments()[0];
	auto &type = state.GetReturnType();
	if (type.id() == LogicalTypeId::SQLNULL) {
		return ReturnExpression(input, state.Copy());
	}
	if (type.IsAggregateState()) {
		vector<unique_ptr<Expression>> arguments;
		arguments.push_back(state.Copy());
		return ReturnExpression(input, BindNative(input.GetClientContext(), "finalize", std::move(arguments)));
	}
	if (!IsCompositeState(type)) {
		throw BinderException("yardstick_finalize requires a native aggregate state or yardstick_state");
	}
	ValidateFields(type);
	auto formula = type.GetAlias().substr(string(STATE_ALIAS_PREFIX).size());
	auto result = BindFormula(input.GetClientContext(), state, ParseFormula(formula));
	auto null_value = make_uniq<BoundConstantExpression>(Value(result->GetReturnType()));
	return ReturnExpression(input, IfNull(state, std::move(null_value), std::move(result)));
}

unique_ptr<FunctionData> BindCombine(BindScalarFunctionInput &input) {
	auto &left = *input.GetArguments()[0];
	auto &right = *input.GetArguments()[1];
	auto &left_type = left.GetReturnType();
	auto &right_type = right.GetReturnType();
	if (left_type.id() == LogicalTypeId::SQLNULL && right_type.id() == LogicalTypeId::SQLNULL) {
		return ReturnExpression(input, left.Copy());
	}
	auto &type = left_type.id() == LogicalTypeId::SQLNULL ? right_type : left_type;
	if (!type.IsAggregateState() && !IsCompositeState(type)) {
		throw BinderException("yardstick_combine requires native aggregate states or yardstick_state values");
	}
	if (left_type.id() == LogicalTypeId::SQLNULL || right_type.id() == LogicalTypeId::SQLNULL) {
		return ReturnExpression(input, left_type.id() == LogicalTypeId::SQLNULL ? right.Copy() : left.Copy());
	}
	if (left_type != right_type) {
		throw BinderException("yardstick_combine requires matching state formulas and aggregate types");
	}
	if (type.IsAggregateState()) {
		vector<unique_ptr<Expression>> arguments;
		arguments.push_back(left.Copy());
		arguments.push_back(right.Copy());
		return ReturnExpression(input, BindNative(input.GetClientContext(), "combine", std::move(arguments)));
	}
	ValidateFields(type);
	vector<unique_ptr<Expression>> fields;
	for (auto &field : StructType::GetChildTypes(type)) {
		vector<unique_ptr<Expression>> arguments;
		arguments.push_back(ExtractState(input.GetClientContext(), left, field.first));
		arguments.push_back(ExtractState(input.GetClientContext(), right, field.first));
		auto combined = BindNative(input.GetClientContext(), "combine", std::move(arguments));
		combined->SetAlias(field.first);
		fields.push_back(std::move(combined));
	}
	auto packed = BindNative(input.GetClientContext(), "struct_pack", std::move(fields));
	auto result = BoundCastExpression::AddCastToType(input.GetClientContext(), std::move(packed), type);
	result = IfNull(right, left.Copy(), std::move(result));
	result = IfNull(left, right.Copy(), std::move(result));
	return ReturnExpression(input, std::move(result));
}

} // namespace

void RegisterYardstickAggregateStateFunctions(ExtensionLoader &loader) {
	ScalarFunction state("yardstick_state", {LogicalType::VARCHAR, LogicalType::ANY}, LogicalType::ANY, nullptr, BindState);
	state.SetBindExpressionCallback(ReplaceStateExpression);
	state.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	loader.RegisterFunction(state);
	ScalarFunction finalize("yardstick_finalize", {LogicalType::ANY}, LogicalType::ANY, nullptr, BindFinalize);
	finalize.SetBindExpressionCallback(ReplaceStateExpression);
	finalize.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	loader.RegisterFunction(finalize);
	ScalarFunction combine("yardstick_combine", {LogicalType::ANY, LogicalType::ANY}, LogicalType::ANY, nullptr, BindCombine);
	combine.SetBindExpressionCallback(ReplaceStateExpression);
	combine.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	loader.RegisterFunction(combine);
}
} // namespace duckdb
#else
namespace duckdb {
void RegisterYardstickAggregateStateFunctions(ExtensionLoader &) {
}
} // namespace duckdb
#endif
