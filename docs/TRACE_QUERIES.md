# Trace query cookbook

Queries against the `agent_traces` table written by `src/agent/traces.py`.
Postgres 12+ (they use `FILTER` and lateral `jsonb_array_elements`).

Set up the table first:

```bash
export TRACE_DSN='postgresql://user:pass@host:5432/db?sslmode=require'
python manage_traces.py init
python manage_traces.py check
```

If you are still on the default JSONL backend, skip to
[Querying JSONL](#querying-jsonl-without-a-database) at the bottom.

## The schema, briefly

Scalar columns are the ones you aggregate over; JSONB columns hold the detail.
That split is deliberate — `avg(cost_usd)` should not have to parse JSON.

| column | notes |
|---|---|
| `trace_id`, `session_id` | One row per investigation. Session groups a run; eval sessions are `eval-<ts>/<arm>/pass<n>`. |
| `created_at` | `TIMESTAMPTZ`, indexed descending. |
| `run_label` | Free text from `AGENT_RUN_LABEL` — git SHA, experiment name, CI build. |
| `provider`, `model`, `prompt_version`, `memory_enabled` | The configuration under test. |
| `case_id` | Eval case name, or `NULL` for ad-hoc investigations. |
| `alert`, `window_start`, `window_end` | The inputs: what the agent was told, and the minute range it was asked about. Enough to re-run an investigation exactly. |
| `stop_reason`, `error` | `completed`, `budget_steps`, `tool_error_limit`, `invalid_output`, `refusal`, `provider_error`. |
| `scope`, `primary_gateway`, `affected_issuer`, `confidence` | The diagnosis, flattened. |
| `steps_count`, `tool_calls`, `tool_errors` | Loop behaviour. |
| `input_tokens`, `output_tokens`, `cache_read_tokens`, `wall_seconds`, `cost_usd` | Cost and latency. |
| `diagnosis`, `steps`, `memory_hits`, `extra` | `JSONB` detail. |

---

## Cost

### Cost and latency per prompt version

The query that justifies the whole trace store. A prompt edit that improves
quality but triples spend should be a visible tradeoff, not a surprise.

```sql
SELECT prompt_version,
       count(*)                                   AS investigations,
       round(avg(cost_usd)::numeric, 4)           AS avg_cost_usd,
       round(sum(cost_usd)::numeric, 2)           AS total_usd,
       round(avg(input_tokens + output_tokens))   AS avg_tokens,
       round(avg(tool_calls)::numeric, 1)         AS avg_tool_calls,
       round(avg(wall_seconds)::numeric, 2)       AS avg_seconds
FROM agent_traces
WHERE provider <> 'baseline-policy'
GROUP BY prompt_version
ORDER BY avg_cost_usd DESC;
```

The `WHERE` matters: the offline baseline policy reports zero tokens, and
pooling it with live runs drags every average toward zero.

### Cost per *successful* diagnosis

Cost per request understates the real number, because failed runs still burn
tokens and have to be retried. This is the figure to quote.

```sql
SELECT prompt_version,
       count(*)                                                       AS attempts,
       count(*) FILTER (WHERE stop_reason = 'completed')               AS completed,
       round(sum(cost_usd)::numeric, 2)                                AS total_usd,
       round((sum(cost_usd)
              / NULLIF(count(*) FILTER (WHERE stop_reason = 'completed'), 0))::numeric,
             4)                                                        AS usd_per_completion
FROM agent_traces
WHERE provider <> 'baseline-policy'
GROUP BY prompt_version;
```

### Daily spend trend

```sql
SELECT date_trunc('day', created_at) AS day,
       count(*)                      AS investigations,
       round(sum(cost_usd)::numeric, 2) AS usd,
       round(avg(wall_seconds)::numeric, 2) AS avg_seconds
FROM agent_traces
WHERE created_at > now() - interval '30 days'
GROUP BY day
ORDER BY day DESC;
```

### Is prompt caching actually working?

`cache_read_tokens` near zero across repeated runs means a silent cache
invalidator — a timestamp in the system prompt, a reordered tool list. Nothing
errors; you just pay full price forever.

```sql
SELECT prompt_version,
       count(*) AS runs,
       round(avg(cache_read_tokens)) AS avg_cached,
       round(avg(input_tokens))      AS avg_input,
       round(100.0 * sum(cache_read_tokens)
             / NULLIF(sum(input_tokens + cache_read_tokens), 0), 1) AS cache_hit_pct
FROM agent_traces
WHERE provider <> 'baseline-policy'
GROUP BY prompt_version;
```

A multi-step investigation resends the whole prefix each step, so anything
below ~50% after the first step deserves investigation.

---

## Reliability

### How do investigations end?

```sql
SELECT stop_reason,
       count(*) AS n,
       round(100.0 * count(*) / sum(count(*)) OVER (), 1) AS pct,
       round(avg(tool_calls)::numeric, 1) AS avg_tool_calls
FROM agent_traces
GROUP BY stop_reason
ORDER BY n DESC;
```

Anything other than `completed` is a loop that gave up. `budget_steps` climbing
means the step ceiling is too tight for the task; `invalid_output` means the
model is fighting the schema; `provider_error` is an outage or a bad key.

### Error rate over time, by day

```sql
SELECT date_trunc('day', created_at) AS day,
       count(*) AS runs,
       round(100.0 * count(*) FILTER (WHERE stop_reason <> 'completed')
             / count(*), 1) AS failure_pct,
       round(100.0 * sum(tool_errors) / NULLIF(sum(tool_calls), 0), 1) AS tool_error_pct
FROM agent_traces
GROUP BY day
ORDER BY day DESC;
```

### Recent failures, with the error text

```sql
SELECT created_at, case_id, stop_reason, left(error, 120) AS error, tool_calls
FROM agent_traces
WHERE stop_reason <> 'completed'
ORDER BY created_at DESC
LIMIT 20;
```

---

## Tool behaviour

### Which tools get called, how slow, how often wrong

Unnests two levels of JSONB: steps, then the tool calls inside each step.

```sql
SELECT call->>'name'                                    AS tool,
       count(*)                                         AS calls,
       count(*) FILTER (WHERE (call->>'is_error')::boolean) AS errors,
       round(avg((call->>'duration_ms')::numeric), 2)   AS avg_ms,
       round(max((call->>'duration_ms')::numeric), 2)   AS max_ms
FROM agent_traces t
CROSS JOIN LATERAL jsonb_array_elements(t.steps) AS step
CROSS JOIN LATERAL jsonb_array_elements(step->'tool_calls') AS call
GROUP BY tool
ORDER BY calls DESC;
```

### Did the agent segment? (a process metric, not an outcome)

An agent that never calls `segment_failures` cannot be *solving* the
issuer-scoped cases, even on the runs where it happens to guess right. Process
metrics catch that; accuracy alone does not.

```sql
SELECT prompt_version,
       count(*) AS runs,
       round(100.0 * count(*) FILTER (
           WHERE EXISTS (
               SELECT 1
               FROM jsonb_array_elements(steps) s,
                    jsonb_array_elements(s->'tool_calls') c
               WHERE c->>'name' = 'segment_failures')
       ) / count(*), 1) AS segmented_pct
FROM agent_traces
GROUP BY prompt_version;
```

### Which tool arguments were malformed

```sql
SELECT call->>'name' AS tool,
       call->'arguments' AS arguments,
       count(*) AS times
FROM agent_traces t
CROSS JOIN LATERAL jsonb_array_elements(t.steps) AS step
CROSS JOIN LATERAL jsonb_array_elements(step->'tool_calls') AS call
WHERE (call->>'is_error')::boolean
GROUP BY tool, arguments
ORDER BY times DESC
LIMIT 20;
```

Repeated identical bad arguments usually means a tool *description* problem,
not a model problem — fix the schema wording before touching the prompt.

---

## Quality

### Diagnosis distribution, and confidence by scope

```sql
SELECT scope,
       count(*)                            AS n,
       round(avg(confidence)::numeric, 3)  AS avg_confidence,
       round(avg(tool_calls)::numeric, 1)  AS avg_tool_calls
FROM agent_traces
WHERE scope IS NOT NULL
GROUP BY scope
ORDER BY n DESC;
```

If `no_incident` is rare, the agent may be inventing incidents when asked to
find one — a false-positive rate that accuracy on incident-bearing cases hides.

### Accuracy against ground truth, for eval runs

Eval traces carry `case_id`, so they can be joined to the expected answer. The
expected values are in `src/agent/evals/cases.py`; inlined here as a CTE so the
query is self-contained.

```sql
WITH truth(case_id, scope, gateway, issuer) AS (VALUES
    ('hard_outage',             'single_gateway',  'PG-Bravo',   NULL),
    ('partial_degradation',     'single_gateway',  'PG-Charlie', NULL),
    ('issuer_outage',           'issuer_specific', 'PG-Delta',   'HDFC'),
    ('issuer_outage_secondary', 'issuer_specific', 'PG-Alpha',   'ICICI'),
    ('fleet_wide',              'fleet_wide',      NULL,         NULL),
    ('no_incident',             'no_incident',     NULL,         NULL),
    ('diurnal_trough',          'no_incident',     NULL,         NULL),
    ('fleet_wide_partial',      'fleet_wide',      NULL,         NULL)
)
SELECT t.prompt_version,
       t.case_id,
       count(*) AS runs,
       round(100.0 * count(*) FILTER (
           WHERE t.scope = tr.scope
             AND t.primary_gateway IS NOT DISTINCT FROM tr.gateway
             AND t.affected_issuer IS NOT DISTINCT FROM tr.issuer
       ) / count(*), 0) AS exact_match_pct
FROM agent_traces t
JOIN truth tr USING (case_id)
GROUP BY t.prompt_version, t.case_id
ORDER BY t.case_id, t.prompt_version;
```

`IS NOT DISTINCT FROM` rather than `=` because both sides are nullable, and
`NULL = NULL` is `NULL`, which would silently score every fleet-wide case wrong.

### Calibration: is confidence earned?

Confidently wrong is the expensive failure on call. This buckets confidence and
shows accuracy in each bucket — a well-calibrated agent's two columns track.

```sql
WITH truth(case_id, scope, gateway, issuer) AS (VALUES
    ('hard_outage','single_gateway','PG-Bravo',NULL),
    ('partial_degradation','single_gateway','PG-Charlie',NULL),
    ('issuer_outage','issuer_specific','PG-Delta','HDFC'),
    ('issuer_outage_secondary','issuer_specific','PG-Alpha','ICICI'),
    ('fleet_wide','fleet_wide',NULL,NULL),
    ('no_incident','no_incident',NULL,NULL),
    ('diurnal_trough','no_incident',NULL,NULL),
    ('fleet_wide_partial','fleet_wide',NULL,NULL)
)
SELECT width_bucket(t.confidence, 0, 1, 5) AS confidence_bucket,
       count(*) AS n,
       round(avg(t.confidence)::numeric, 2) AS stated_confidence,
       round(avg(CASE WHEN t.scope = tr.scope
                       AND t.primary_gateway IS NOT DISTINCT FROM tr.gateway
                       AND t.affected_issuer IS NOT DISTINCT FROM tr.issuer
                      THEN 1.0 ELSE 0.0 END)::numeric, 2) AS actual_accuracy
FROM agent_traces t
JOIN truth tr USING (case_id)
GROUP BY confidence_bucket
ORDER BY confidence_bucket;
```

---

## Trace → eval case

The pipeline this store exists to enable: find investigations that look wrong,
then promote them into regression tests.

```sql
SELECT trace_id, created_at, case_id, scope, primary_gateway,
       confidence, tool_calls, stop_reason
FROM agent_traces
WHERE stop_reason <> 'completed'
   OR confidence < 0.5
   OR tool_calls >= 12          -- flailing
   OR tool_errors > 0
ORDER BY created_at DESC
LIMIT 50;
```

Pull one apart in full:

```sql
SELECT jsonb_pretty(steps) FROM agent_traces WHERE trace_id = '<paste-id>';
```

That prints every tool call with its arguments and the exact result the agent
saw — which is the only reliable way to see where a diagnosis turned wrong.

---

## Housekeeping

### Rows and size

```sql
SELECT count(*) AS rows,
       pg_size_pretty(pg_total_relation_size('agent_traces')) AS size,
       min(created_at) AS oldest,
       max(created_at) AS newest
FROM agent_traces;
```

Tool results are the bulk of it. If rows get large, set
`TRACE_INCLUDE_TOOL_RESULTS=false` to keep arguments but drop results.

### Retention

There is no automatic retention — a trace store that grows forever is the
common failure. Decide a window and enforce it:

```sql
DELETE FROM agent_traces WHERE created_at < now() - interval '90 days';
```

Keep the aggregate first if you want history without the bulk:

```sql
CREATE TABLE IF NOT EXISTS agent_traces_daily AS
SELECT date_trunc('day', created_at) AS day, prompt_version, model,
       count(*) AS runs,
       sum(cost_usd) AS usd,
       avg(wall_seconds) AS avg_seconds,
       count(*) FILTER (WHERE stop_reason <> 'completed') AS failures
FROM agent_traces
GROUP BY 1, 2, 3;
```

### Clean up probe rows

`manage_traces.py check` writes a marker row each time it runs:

```sql
DELETE FROM agent_traces WHERE session_id = 'connectivity-check';
```

---

## Querying JSONL without a database

The default backend appends one JSON object per line, so ad-hoc analysis needs
no database at all.

```bash
python manage_traces.py tail -n 20
```

```bash
# total cost by prompt version
jq -s 'group_by(.prompt_version)[]
       | {prompt: .[0].prompt_version, runs: length,
          usd: (map(.cost_usd) | add)}' results/traces.jsonl
```

```bash
# stop-reason distribution
jq -r '.stop_reason' results/traces.jsonl | sort | uniq -c | sort -rn
```

Without `jq`:

```bash
python -c "
import json, collections
rows=[json.loads(l) for l in open('results/traces.jsonl')]
print(collections.Counter(r['stop_reason'] for r in rows))
print('total usd:', round(sum(r['cost_usd'] for r in rows), 4))
"
```

The JSONL field names match the Postgres columns exactly, so a query written
against one translates directly to the other.
