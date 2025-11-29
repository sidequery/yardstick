# Yardstick: Known Limitations

Implementation of Julian Hyde's "Measures in SQL" paper (arXiv:2406.00251).

## Working Features

- `AS MEASURE` in CREATE VIEW to define measures
- `AGGREGATE(measure)` function to evaluate measures
- `AT (ALL)` - grand total (removes all dimensions)
- `AT (ALL dim)` - removes specific dimension from context
- `AT (SET dim = expr)` - fixes dimension to specific value (e.g., YoY comparisons)
- `AT (WHERE condition)` - filters before aggregation
- `AT (VISIBLE)` - uses visible WHERE clause filters
- Multiple measures in same view
- Arithmetic with AGGREGATE results (ratios, percentages, differences)
- SUM, COUNT, MIN, MAX, AVG aggregations

## Known Limitations

### 1. Chained AT Modifiers

```sql
-- ISSUE: Chaining multiple AT modifiers doesn't work as expected
AGGREGATE(revenue) AT (ALL region) AT (ALL category)
-- Returns grand total instead of year-only total
-- Expected: Remove region, then remove category (keeping year)
-- Actual: Removes all dimensions
```

**Workaround**: Use single `AT (ALL dim1, dim2)` syntax if supported, or restructure query.

### 2. CASE Expressions in Measures

```sql
-- NOT SUPPORTED
CREATE VIEW v AS
SELECT year, CASE WHEN SUM(x) > 100 THEN 1 ELSE 0 END AS MEASURE flag
FROM t GROUP BY year;
-- Error: Measure not found
```

**Workaround**: Create the CASE expression outside the measure, or use a subquery.

### 3. Table Aliases with Qualified References

```sql
-- ISSUE: Table aliases conflict with internal _outer alias
SELECT s.year, AGGREGATE(s.revenue) FROM sales_v s GROUP BY s.year;
-- May produce incorrect results or errors
```

**Workaround**: Don't use table aliases, or don't use qualified column references.

### 4. No Derived Measures

```sql
-- NOT YET SUPPORTED: Measures referencing other measures
CREATE VIEW v AS
SELECT year,
  SUM(revenue) AS MEASURE revenue,
  SUM(cost) AS MEASURE cost,
  revenue - cost AS MEASURE profit  -- NOT SUPPORTED
FROM t GROUP BY year;
```

**Workaround**: Calculate derived values in the SELECT using AGGREGATE().

### 5. No Window Function Measures

```sql
-- NOT SUPPORTED: Window functions in measure definitions
CREATE VIEW v AS
SELECT year,
  SUM(revenue) OVER (ORDER BY year) AS MEASURE running_total
FROM t;
```

## AT Modifier Reference

| Modifier | Description | Status |
|----------|-------------|--------|
| `AT (ALL)` | Grand total, removes all dimensions | Working |
| `AT (ALL dim)` | Removes specific dimension | Working |
| `AT (ALL dim1, dim2)` | Removes multiple dimensions | Working |
| `AT (SET dim = val)` | Fixes dimension to value | Working |
| `AT (WHERE cond)` | Pre-aggregation filter | Working |
| `AT (VISIBLE)` | Uses query's WHERE clause | Working |
| Chained AT | Multiple AT modifiers | Limited |

## Implementation Notes

The implementation rewrites AGGREGATE() calls into correlated subqueries:

```sql
-- Input:
SELECT year, AGGREGATE(revenue) AT (ALL year) FROM v GROUP BY year;

-- Rewritten to:
SELECT year, (
  SELECT SUM(revenue) FROM v AS _outer
  WHERE TRUE  -- AT (ALL year) removes year filter
) FROM v GROUP BY year;
```

This approach has limitations with complex expressions and nested contexts.
