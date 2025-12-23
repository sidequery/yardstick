# Yardstick: Known Limitations

Implementation of Julian Hyde's "Measures in SQL" paper (arXiv:2406.00251).

## Working Features

- `AS MEASURE` in CREATE VIEW to define measures
- `AGGREGATE(measure)` function to evaluate measures
- `AT (ALL)` - grand total (removes all dimensions)
- `AT (ALL dim)` - removes specific dimension from context
- Chained AT modifiers: `AT (ALL dim1) AT (ALL dim2)` correctly correlates on remaining dimensions
- `AT (SET dim = expr)` - fixes dimension to specific value (e.g., YoY comparisons)
- Ad hoc dimensions: `AT (ALL MONTH(date))` and `AT (SET MONTH(date) = 6)` for computed dimensions
- `AT (WHERE condition)` - filters before aggregation
- `AT (VISIBLE)` - uses visible WHERE clause filters
- Multiple measures in same view
- Arithmetic with AGGREGATE results (ratios, percentages, differences)
- All DuckDB aggregate functions (SUM, COUNT, AVG, MIN, MAX, STDDEV, MEDIAN, etc.)
- COUNT(DISTINCT) aggregations (with restrictions, see below)
- Derived measures: `revenue - cost AS MEASURE profit` expands to `SUM(revenue) - SUM(cost)`
- Multi-fact JOINs: measures from different views can be queried together in a single JOIN

## Known Limitations

### 1. Non-Decomposable Aggregates (MEDIAN, PERCENTILE, MODE)

```sql
-- NOT SUPPORTED with AGGREGATE(): MEDIAN, PERCENTILE, MODE
CREATE VIEW v AS
SELECT region, MEDIAN(amount) AS MEASURE median_amount
FROM orders;

-- Direct query works fine:
SELECT region, median_amount FROM v;

-- But AGGREGATE() fails (cannot re-aggregate):
SEMANTIC SELECT region, AGGREGATE(median_amount) FROM v;  -- ERROR
```

Non-decomposable aggregates like MEDIAN and PERCENTILE cannot be re-aggregated. Query these views directly without AGGREGATE(). Note: COUNT(DISTINCT) is handled specially, see below.

### 2. No Window Function Measures

```sql
-- NOT SUPPORTED: Window functions in measure definitions
CREATE VIEW v AS
SELECT year,
  SUM(revenue) OVER (ORDER BY year) AS MEASURE running_total
FROM t;
```

### 3. COUNT(DISTINCT) AT Modifier Restrictions

COUNT(DISTINCT) is non-decomposable: you cannot re-aggregate distinct counts from subsets without double-counting. Yardstick handles this by evaluating COUNT(DISTINCT) via correlated subqueries at query time.

**Works:**
```sql
-- Basic COUNT(DISTINCT) measure
CREATE VIEW orders_v AS
SELECT year, region, COUNT(DISTINCT customer_id) AS MEASURE unique_customers
FROM orders;

-- Simple aggregation
SEMANTIC SELECT year, region, AGGREGATE(unique_customers) FROM orders_v;

-- AT (WHERE) - just adds a filter
SEMANTIC SELECT year, AGGREGATE(unique_customers) AT (WHERE region = 'US') FROM orders_v;

-- AT (VISIBLE) - applies outer WHERE clause
SEMANTIC SELECT year, AGGREGATE(unique_customers) AT (VISIBLE) FROM orders_v WHERE region = 'US';
```

**Does NOT work:**
```sql
-- AT (ALL) - grand total requires re-aggregating distinct counts
AGGREGATE(unique_customers) AT (ALL)

-- AT (ALL dim) - removing a dimension requires re-aggregation
AGGREGATE(unique_customers) AT (ALL region)

-- AT (SET) - comparing across contexts requires re-aggregation
AGGREGATE(unique_customers) AT (SET year = year - 1)
```

These will return an error explaining that the modifier is not supported for non-decomposable measures.

