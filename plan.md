# Plan: Cache Attribution Graphs in Database

## Overview
Add database caching for computed attribution graphs to avoid recomputation for the same prompts. Store raw (unnormalized) edges and apply normalization on retrieval.

## Cache Key Strategy

**Standard graphs:** `(run_id, token_ids_hash)`
- Token IDs uniquely identify the prompt within a run

**Optimized graphs:** `(run_id, token_ids_hash, label_token, imp_min_coeff, ce_loss_coeff, steps, pnorm)`
- All optimization parameters affect the result

**Normalization:** Not part of cache key - store raw edges, normalize on retrieval

## Database Schema Changes

Add new table in `spd/app/backend/db/database.py`:

```sql
CREATE TABLE IF NOT EXISTS cached_graphs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    token_ids_hash TEXT NOT NULL,        -- SHA256 of JSON-encoded token_ids
    token_ids TEXT NOT NULL,             -- JSON array of token IDs
    is_optimized INTEGER NOT NULL,       -- 0 = standard, 1 = optimized

    -- Optimization params (NULL for standard graphs)
    label_token INTEGER,
    imp_min_coeff REAL,
    ce_loss_coeff REAL,
    steps INTEGER,
    pnorm REAL,

    -- Cached data (gzipped JSON)
    edges_data BLOB NOT NULL,            -- list of {src, tgt, val, is_cross_seq}
    output_probs_data BLOB NOT NULL,     -- dict of {pos:cIdx -> {prob, token}}

    -- Optimization stats (NULL for standard graphs)
    label_prob REAL,
    l0_total REAL,
    l0_per_layer TEXT,                   -- JSON dict

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(run_id, token_ids_hash, is_optimized, label_token, imp_min_coeff, ce_loss_coeff, steps, pnorm)
);

CREATE INDEX IF NOT EXISTS idx_cached_graphs_lookup
    ON cached_graphs(run_id, token_ids_hash, is_optimized);
```

## Implementation Steps

### 1. Database layer (`db/database.py`)
- Add `save_graph()` method to store computed graph
- Add `get_cached_graph()` method to retrieve by cache key
- Add `delete_cached_graphs()` method to clear cache for a run
- Use gzip compression for edges/output_probs (similar to activation_contexts)

### 2. Backend routes (`routers/graphs.py`)
- Before computation: check cache with `get_cached_graph()`
- If hit: return cached data immediately (skip streaming progress)
- If miss: compute as normal, then `save_graph()` before returning
- Apply normalization after retrieving from cache (not stored)

### 3. API response changes
- Add `cached: boolean` field to response so frontend knows it was cached
- Consider adding cache stats endpoint (optional)

### 4. Frontend changes (minimal)
- Show indicator when graph was loaded from cache (optional)
- Could show "cached" badge on graph tab label

## Files to Modify

1. `spd/app/backend/db/database.py` - Add cache table and methods
2. `spd/app/backend/routers/graphs.py` - Add cache lookup/save logic
3. `spd/app/backend/schemas.py` - Add `cached` field to GraphData (optional)
4. `spd/app/frontend/src/lib/localAttributionsTypes.ts` - Add `cached` field (optional)

## Edge Cases

- **Cache invalidation:** Currently no invalidation needed (same run + same params = same result)
- **DB migrations:** Use `IF NOT EXISTS` so existing DBs get the new table
- **Large graphs:** Gzip compression should keep size manageable
- **Token ID ordering:** Hash should be stable (JSON array maintains order)

## Testing

- Compute graph, verify it's cached
- Compute same graph again, verify cache hit (no progress streaming)
- Compute with different normalize param, verify same cache entry used
- Compute optimized with different params, verify separate cache entries
