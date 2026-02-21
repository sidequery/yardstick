/**
 * yardstick_ffi.h - C FFI for DuckDB parser integration
 *
 * Flat C structs for passing data between C++ (DuckDB) and Rust (yardstick-rs).
 * All memory is allocated by Rust and freed via explicit free functions.
 */

#ifndef YARDSTICK_FFI_H
#define YARDSTICK_FFI_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* =============================================================================
 * AT Modifier Types (matches ContextModifier enum in Rust)
 * ============================================================================= */

typedef enum {
    YARDSTICK_AT_NONE = 0,      /* No modifier */
    YARDSTICK_AT_ALL_GLOBAL,    /* AT (ALL) - grand total */
    YARDSTICK_AT_ALL_DIM,       /* AT (ALL dimension) */
    YARDSTICK_AT_SET,           /* AT (SET dimension = expr) */
    YARDSTICK_AT_WHERE,         /* AT (WHERE condition) */
    YARDSTICK_AT_VISIBLE,       /* AT (VISIBLE) */
} YardstickAtType;

/* Single AT modifier */
typedef struct {
    YardstickAtType type;
    const char* dimension;      /* For ALL_DIM, SET: dimension name (NULL otherwise) */
    const char* value;          /* For SET: expression value; for WHERE: condition */
} YardstickAtModifier;

/* =============================================================================
 * Aggregate Call Information
 * ============================================================================= */

/* Information about a single AGGREGATE() call in SQL */
typedef struct {
    const char* measure_name;   /* e.g., "revenue" */
    uint32_t start_pos;         /* byte position in original SQL */
    uint32_t end_pos;           /* byte position (exclusive) */

    /* AT modifier chain (supports multiple: AGGREGATE(x) AT (...) AT (...)) */
    YardstickAtModifier* modifiers;
    size_t modifier_count;
} YardstickAggregateCall;

/* List of all AGGREGATE() calls found in SQL */
typedef struct {
    YardstickAggregateCall* calls;
    size_t count;
    const char* error;          /* NULL if success */
} YardstickAggregateCallList;

/* =============================================================================
 * SELECT Clause Information
 * ============================================================================= */

/* Information about a single SELECT item */
typedef struct {
    const char* expression_sql; /* Raw SQL text, e.g., "SUM(amount)" */
    const char* alias;          /* Alias if present, NULL otherwise */
    uint32_t start_pos;         /* byte position in original SQL */
    uint32_t end_pos;           /* byte position (exclusive) */
    bool is_aggregate;          /* Contains SUM/COUNT/AVG/MIN/MAX */
    bool is_star;               /* True if SELECT * or table.* */
    bool is_measure_ref;        /* True if references AGGREGATE() */
} YardstickSelectItem;

/* Information about FROM clause tables */
typedef struct {
    const char* table_name;     /* Original table name */
    const char* alias;          /* Alias if present (NULL = same as table_name) */
    bool is_subquery;           /* True if derived table */
} YardstickTableRef;

/* Full SELECT clause information */
typedef struct {
    YardstickSelectItem* items;
    size_t item_count;

    YardstickTableRef* tables;  /* All tables including JOINs */
    size_t table_count;
    const char* primary_table;  /* First table in FROM (convenience) */

    /* GROUP BY info */
    const char** group_by_cols; /* Column expressions in GROUP BY */
    size_t group_by_count;
    bool has_group_by;
    bool group_by_all;          /* True if GROUP BY ALL */

    /* WHERE clause */
    const char* where_clause;   /* NULL if none */

    const char* error;          /* NULL if success */
} YardstickSelectInfo;

/* =============================================================================
 * Expression Parse Result
 * ============================================================================= */

/* Parsed expression information */
typedef struct {
    const char* sql;            /* Normalized SQL text */
    const char* aggregate_func; /* "SUM", "COUNT", etc. or NULL if not aggregate */
    const char* inner_expr;     /* For SUM(x), this is "x"; NULL otherwise */
    bool is_aggregate;
    bool is_identifier;         /* True if just a column reference */
    const char* error;          /* NULL if success */
} YardstickExpressionInfo;

/* =============================================================================
 * Measure Definition (for CREATE VIEW AS MEASURE)
 * ============================================================================= */

typedef struct {
    const char* column_name;    /* Measure name, e.g., "revenue" */
    const char* expression;     /* Measure expression, e.g., "SUM(amount)" */
    const char* aggregate_func; /* Extracted agg function or NULL for derived */
    bool is_derived;            /* True if references other measures */
} YardstickMeasureDef;

/* Result from parsing CREATE VIEW with AS MEASURE */
typedef struct {
    bool is_measure_view;
    const char* view_name;
    const char* clean_sql;      /* SQL with AS MEASURE removed */
    YardstickMeasureDef* measures;
    size_t measure_count;
    const char* error;
} YardstickCreateViewInfo;

/* =============================================================================
 * SQL Replacement Operation
 * ============================================================================= */

/* Single replacement in SQL text */
typedef struct {
    uint32_t start_pos;
    uint32_t end_pos;
    const char* replacement;
} YardstickReplacement;

/* =============================================================================
 * FFI Functions - Aggregate Detection
 * ============================================================================= */

/**
 * Find all AGGREGATE() calls in SQL, including AT modifiers.
 *
 * @param sql  The SQL string to parse
 * @return Allocated list of aggregate calls. Caller must free with
 *         yardstick_free_aggregate_list(). Check error field for failures.
 */
YardstickAggregateCallList* yardstick_find_aggregates(const char* sql);

/**
 * Free an aggregate call list returned by yardstick_find_aggregates().
 */
void yardstick_free_aggregate_list(YardstickAggregateCallList* list);

/* =============================================================================
 * FFI Functions - SELECT Analysis
 * ============================================================================= */

/**
 * Parse SELECT statement and extract clause information.
 *
 * @param sql  The SQL SELECT statement
 * @return Allocated select info. Caller must free with yardstick_free_select_info().
 */
YardstickSelectInfo* yardstick_parse_select(const char* sql);

/**
 * Free a select info structure.
 */
void yardstick_free_select_info(YardstickSelectInfo* info);

/* =============================================================================
 * FFI Functions - Expression Parsing
 * ============================================================================= */

/**
 * Parse a single SQL expression (for measure definitions).
 *
 * @param expr  The expression to parse, e.g., "SUM(amount)"
 * @return Allocated expression info. Caller must free with yardstick_free_expression_info().
 */
YardstickExpressionInfo* yardstick_parse_expression(const char* expr);

/**
 * Free an expression info structure.
 */
void yardstick_free_expression_info(YardstickExpressionInfo* info);

/* =============================================================================
 * FFI Functions - CREATE VIEW Processing
 * ============================================================================= */

/**
 * Parse CREATE VIEW with AS MEASURE syntax.
 *
 * @param sql  The CREATE VIEW statement
 * @return Allocated create view info. Caller must free with yardstick_free_create_view_info().
 */
YardstickCreateViewInfo* yardstick_parse_create_view(const char* sql);

/**
 * Free a create view info structure.
 */
void yardstick_free_create_view_info(YardstickCreateViewInfo* info);

/* =============================================================================
 * FFI Functions - SQL Rewriting
 * ============================================================================= */

/**
 * Apply multiple replacements to SQL (handles position adjustments).
 * Replacements are applied in order; positions refer to original SQL.
 *
 * @param sql           Original SQL string
 * @param replacements  Array of replacement operations
 * @param count         Number of replacements
 * @return Allocated string with replacements applied. Caller must free with
 *         yardstick_free_string().
 */
char* yardstick_apply_replacements(
    const char* sql,
    const YardstickReplacement* replacements,
    size_t count
);

/**
 * Replace a single range in SQL.
 *
 * @param sql         Original SQL string
 * @param start       Start byte position
 * @param end         End byte position (exclusive)
 * @param replacement Replacement text
 * @return Allocated string with replacement. Caller must free with yardstick_free_string().
 */
char* yardstick_replace_range(
    const char* sql,
    uint32_t start,
    uint32_t end,
    const char* replacement
);

char* yardstick_qualify_expression(const char* expr, const char* qualifier);

/**
 * Free a string allocated by yardstick functions.
 */
void yardstick_free_string(char* ptr);

/* =============================================================================
 * FFI Functions - Measure Expansion
 * ============================================================================= */

/**
 * Expand a single AGGREGATE() call to SQL.
 *
 * @param measure_name   Name of the measure being aggregated
 * @param agg_func       Aggregation function: "SUM", "COUNT", etc.
 * @param modifiers      Array of AT modifiers (can be NULL)
 * @param modifier_count Number of modifiers
 * @param table_name     Source table name
 * @param outer_alias    Outer query alias (can be NULL)
 * @param outer_where    Outer query WHERE clause (can be NULL)
 * @param group_by_cols  Array of GROUP BY column names (can be NULL)
 * @param group_by_count Number of GROUP BY columns
 * @return Allocated string with expanded SQL. Caller must free with yardstick_free_string().
 */
char* yardstick_expand_aggregate_call(
    const char* measure_name,
    const char* agg_func,
    const YardstickAtModifier* modifiers,
    size_t modifier_count,
    const char* table_name,
    const char* outer_alias,
    const char* outer_where,
    const char* const* group_by_cols,
    size_t group_by_count
);

/* =============================================================================
 * Existing FFI Functions (backward compatibility)
 * ============================================================================= */

/* These match the current ffi.rs interface */

struct YardstickCreateViewResult {
    bool is_measure_view;
    char* view_name;
    char* clean_sql;
    char* error;
};

struct YardstickAggregateResult {
    bool had_aggregate;
    char* expanded_sql;
    char* error;
};

struct YardstickMeasureAggResult {
    char* aggregation;
    char* view_name;
};

bool yardstick_has_as_measure(const char* sql);
bool yardstick_has_aggregate(const char* sql);
bool yardstick_drop_measure_view_from_sql(const char* sql);
bool yardstick_has_curly_brace(const char* sql);
bool yardstick_has_at_syntax(const char* sql);

char* yardstick_expand_curly_braces(const char* sql);
struct YardstickCreateViewResult yardstick_process_create_view(const char* sql);
struct YardstickAggregateResult yardstick_expand_aggregate(const char* sql);
struct YardstickMeasureAggResult yardstick_get_measure_aggregation(const char* column_name);

void yardstick_free(char* ptr);
void yardstick_free_create_view_result(struct YardstickCreateViewResult result);
void yardstick_free_aggregate_result(struct YardstickAggregateResult result);
void yardstick_free_measure_agg_result(struct YardstickMeasureAggResult result);

#ifdef __cplusplus
}
#endif

#endif /* YARDSTICK_FFI_H */
