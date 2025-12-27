# Ordered-Set / Non-Decomposable Aggregates in Yardstick

This is a design note for supporting MEDIAN, PERCENTILE_CONT/PERCENTILE_DISC, QUANTILE/QUANTILE_CONT/QUANTILE_DISC, and MODE with AGGREGATE() in Yardstick.

The core problem is non-decomposability: these aggregates cannot be re-aggregated from partial results. COUNT(DISTINCT) is already handled by recomputing from base rows. The proposal below extends that approach with ordered-set semantics and a join-based optimization path.

## Goals

- Correct results for ordered-set aggregates under Yardstick semantics.
- Support a reasonable subset of AT modifiers without violating correctness.
- Keep behavior predictable and transparent for users.
- Avoid new SQL features or custom kernels in DuckDB if possible.

Non-goals:
- Perfect performance for all queries. The recompute path is expected to be more expensive.
- Supporting windowed or custom aggregates.

## Aggregate Coverage

- MEDIAN(x): treat as PERCENTILE_CONT(0.5) with a specific null/tie behavior (DuckDB semantics).
- PERCENTILE_CONT(q) and PERCENTILE_DISC(q): ordered-set aggregates with a required ORDER BY argument in standard SQL, but DuckDB accepts inline argument styles. We should treat the view expression as authoritative and preserve its semantics.
- QUANTILE / QUANTILE_CONT / QUANTILE_DISC: DuckDB-specific variants similar to percentiles.
- MODE: non-decomposable; tie behavior is engine-defined.

## AT Modifier Support: What Is Reasonable?

### Always reasonable (recompute from base rows)

These modifiers do not require re-aggregating partial aggregates; they simply change the input row set or context. They can be implemented by recompute with correlated filters.

- AT (WHERE cond)
- AT (VISIBLE)

### Reasonable but potentially expensive

These modifiers change the correlation context and require recomputation from the base relation for each outer group.

- AT (ALL) (global recompute)
- AT (ALL dim) (recompute on remaining dimensions)
- AT (SET dim = expr) (recompute in altered context)

### Not reasonable (initially)

- Chained AT modifiers beyond simple combinations (e.g., AT (ALL dim) AT (SET ...) with ad hoc dimensions) could be supported but needs careful semantics. This should be explicitly deferred if we can’t define correctness clearly.
- Ad hoc dimensions inside ordered-set expressions (e.g., MEDIAN(price * fx_rate)) are fine, but ad hoc dimensions inside AT (SET expr = ...) for expressions used in ORDER BY could be ambiguous.

**Recommendation**: Support the same AT modifier surface as COUNT(DISTINCT), but document that ordered-set recompute is expensive. If any modifier combination causes ambiguous correlation, return a clear error.

## Semantics and Rewrite Strategy

### 1) Base-relation recompute (correct, simplest)

For a non-decomposable measure defined as:

```
SELECT
  dim1,
  MEDIAN(x) AS MEASURE med_x
FROM base
```

Rewrite:

```
SEMANTIC SELECT dim1, AGGREGATE(med_x) FROM v
```

To:

```
SELECT dim1,
  (SELECT MEDIAN(_inner.x)
   FROM (base_relation) _inner
   WHERE _inner.dim1 = _outer.dim1
  ) AS med_x
FROM v _outer
```

AT modifier handling parallels COUNT(DISTINCT):

- AT (ALL): drop correlation, aggregate over full base relation.
- AT (ALL dim): correlate on remaining dimensions.
- AT (SET dim = expr): replace outer correlation for that dim with a computed value.
- AT (WHERE cond): add an inner WHERE filter before aggregation.
- AT (VISIBLE): propagate outer WHERE into inner filter.

This yields correct results but can be expensive on large grouped queries.

### 2) Join-based recompute (optimization)

When the outer query groups by dimensions, we can precompute the ordered-set aggregate once per group and join it back, avoiding per-row correlated subqueries.

Example:

```
SELECT year, AGGREGATE(med_x) FROM v GROUP BY year
```

Rewrite to:

```
SELECT _outer.year, _nd.value AS med_x
FROM v _outer
LEFT JOIN (
  SELECT _inner.year AS dim_0,
         MEDIAN(_inner.x) AS value
  FROM (base_relation) _inner
  GROUP BY _inner.year
) _nd
ON _nd.dim_0 = _outer.year
```

This is the same strategy currently used for COUNT(DISTINCT).

We can extend the existing non-decomposable join plan to support ordered-set aggregates by reusing the view’s exact expression, preserving semantics.

## Extracting Ordered-Set Semantics

The measure expression is already stored verbatim (e.g., `MEDIAN(amount)` or `PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY amount)`).

The recompute path should reuse the full expression in the inner aggregate. We should **not** try to normalize or re-parse the ordered-set syntax; instead, pass it through to DuckDB unchanged, except for column qualification in the inner query.

We already have:
- `qualify_where_for_inner` and `qualify_where_for_inner_with_dimensions`.
- `dimension_exprs` for correlating ad hoc dimensions.

We will need:
- A safer qualifier for aggregate expressions that may contain ORDER BY and/or FILTER clauses, to avoid qualifying function names or keywords. (Potentially reusing the existing qualifier with cautious tokenization.)

## Proposed Implementation Steps

1. **Detect ordered-set aggregates**
   - Extend `NON_DECOMPOSABLE_AGGREGATES` detection to include MEDIAN, PERCENTILE_*, QUANTILE_*, MODE.
   - Mark these as `is_decomposable = false` (same as COUNT DISTINCT).

2. **Reuse existing non-decomposable recompute pipeline**
   - The same non-decomposable expansion logic can handle any aggregate expression if it is passed through and correctly qualified.
   - Ensure the expression is used verbatim in inner aggregate (except column qualification).

3. **Qualification of expression**
   - Implement `qualify_aggregate_expr_for_inner(expr)` that only prefixes column references, not keywords or function names.
   - Reuse in join plan and correlated subqueries.

4. **AT modifier support**
   - Use the same modifier logic as COUNT(DISTINCT).
   - If correlation fails (e.g., missing GROUP BY dims), fall back to correlated subquery.

5. **CTE handling**
   - Use the existing base relation extraction logic (`WITH ... SELECT * FROM <from>`) so recompute includes CTE filters and sources.

## Edge Cases

- **NULL handling**: rely on DuckDB aggregate semantics. If needed, document explicitly that behavior matches DuckDB.
- **MODE ties**: DuckDB’s MODE tie behavior should be treated as authoritative.
- **Multiple ORDER BY terms** (if supported): pass through as-is.
- **FILTER clause in measure**: ensure filter stays inside the inner aggregate. Avoid double-filtering when AT (WHERE) is also used.
- **Outer WHERE + AT (VISIBLE)**: ensure filter is applied once, same as current COUNT DISTINCT logic.
- **Ad hoc dimensions**: existing dimension_exprs should map expressions to correlation conditions.

## Tests

SQLLogic tests (examples):

- MEDIAN basic recompute
- MEDIAN with AT (WHERE)
- PERCENTILE_CONT with AT (ALL)
- MODE with AT (SET)
- QUANTILE with CTE-based view

Unit tests:

- qualification of ordered-set expressions
- join-based plan selection for non-decomposable ordered-set aggregates

## Performance Considerations

- Recompute path is potentially expensive on large datasets.
- Join-based plan should be used whenever group-by correlation is possible.
- Consider explicit warnings in docs for ordered-set aggregates similar to COUNT(DISTINCT).

## Open Questions

- Should we accept all AT modifier combinations for ordered-set aggregates, or return errors for ambiguous mixes?
- Do we need explicit opt-in (e.g., feature flag) for ordered-set recompute due to cost?
- Should the docs recommend pre-aggregated materialized views for high-cost cases?

