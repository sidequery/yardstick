# Yardstick

A DuckDB extension implementing Julian Hyde's "Measures in SQL" paper ([arXiv:2406.00251](https://arxiv.org/abs/2406.00251)).

## What is this?

Yardstick adds **measure-aware SQL** to DuckDB. Measures are aggregations that know how to re-aggregate themselves when the query context changes. This enables:

- **Percent of total** calculations without CTEs or window functions
- **Year-over-year comparisons** with simple syntax
- **Drill-down analytics** that automatically adjust aggregation context

## Quick Start

```sql
-- Load the extension
LOAD 'yardstick';

-- Create a view with measures
CREATE VIEW sales_v AS
SELECT
    year,
    region,
    SUM(amount) AS MEASURE revenue,
    COUNT(*) AS MEASURE order_count
FROM sales
GROUP BY year, region;

-- Query with AGGREGATE() and AT modifiers
SELECT
    year,
    region,
    AGGREGATE(revenue) AS revenue,
    AGGREGATE(revenue) AT (ALL region) AS year_total,
    AGGREGATE(revenue) / AGGREGATE(revenue) AT (ALL region) AS pct_of_year
FROM sales_v
GROUP BY year, region;
```

## Syntax

### Defining Measures

```sql
CREATE VIEW view_name AS
SELECT
    dimension1,
    dimension2,
    AGG(expr) AS MEASURE measure_name
FROM table
GROUP BY dimension1, dimension2;
```

Supported aggregations: `SUM`, `COUNT`, `AVG`, `MIN`, `MAX`

### Querying Measures

```sql
SELECT
    dimensions,
    AGGREGATE(measure_name) [AT modifier]
FROM view_name
GROUP BY dimensions;
```

### AT Modifiers

| Modifier | Description | Example |
|----------|-------------|---------|
| `AT (ALL)` | Grand total across all dimensions | `AGGREGATE(revenue) AT (ALL)` |
| `AT (ALL dim)` | Total excluding specific dimension | `AGGREGATE(revenue) AT (ALL region)` |
| `AT (SET dim = val)` | Fix dimension to specific value | `AGGREGATE(revenue) AT (SET year = 2022)` |
| `AT (SET dim = expr)` | Fix dimension to expression | `AGGREGATE(revenue) AT (SET year = year - 1)` |
| `AT (WHERE cond)` | Pre-aggregation filter | `AGGREGATE(revenue) AT (WHERE region = 'US')` |
| `AT (VISIBLE)` | Use query's WHERE clause | `AGGREGATE(revenue) AT (VISIBLE)` |

## Examples

### Percent of Total

```sql
SELECT
    region,
    AGGREGATE(revenue) AS revenue,
    100.0 * AGGREGATE(revenue) / AGGREGATE(revenue) AT (ALL) AS pct_total
FROM sales_v
GROUP BY region;
```

### Year-over-Year Growth

```sql
SELECT
    year,
    AGGREGATE(revenue) AS revenue,
    AGGREGATE(revenue) AT (SET year = year - 1) AS prior_year,
    100.0 * (AGGREGATE(revenue) - AGGREGATE(revenue) AT (SET year = year - 1))
          / AGGREGATE(revenue) AT (SET year = year - 1) AS yoy_growth
FROM sales_v
GROUP BY year;
```

### Contribution to Parent

```sql
SELECT
    year,
    region,
    AGGREGATE(revenue) AS revenue,
    AGGREGATE(revenue) AT (ALL region) AS year_total,
    100.0 * AGGREGATE(revenue) / AGGREGATE(revenue) AT (ALL region) AS contribution
FROM sales_v
GROUP BY year, region;
```

## Building

Prerequisites:
- CMake 3.5+
- C++17 compiler
- Rust (for the SQL rewriter)

```bash
# Build Rust library first
cd yardstick-rs && cargo build --release && cd ..

# Build DuckDB extension
make
```

The extension will be at `build/release/extension/yardstick/yardstick.duckdb_extension`

## Testing

```bash
make test
```

## Limitations

See [LIMITATIONS.md](LIMITATIONS.md) for known issues and workarounds.

Key limitations:
- Chained AT modifiers collapse to grand total instead of removing dimensions sequentially
- Derived measures (measures referencing other measures) not yet supported
- Window function measures not supported

## References

- Julian Hyde, "Measures in SQL" (2024). [arXiv:2406.00251](https://arxiv.org/abs/2406.00251)
- [DuckDB Extension Template](https://github.com/duckdb/extension-template)

## License

MIT
