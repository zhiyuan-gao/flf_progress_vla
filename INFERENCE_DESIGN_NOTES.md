# Inference design notes and deferred decisions

Status: recorded before the formal 1000-step continuation training. These notes separate the
frozen training contract from inference-time choices that will be evaluated after training.

## 1. Training contract is independent of the localizer

Training uses oracle conditions derived from each demonstration segment:

```text
annotation -> oracle stage k_t -> stride-8 grid progress p_t -> state[20:22] -> GR00T
```

The trainer does not run a reference-video localizer, keep a query history, or apply the stage
completion controller. The following are therefore inference-only choices and do not require a
new checkpoint as long as the meanings and normalization of `(k_t, p_t)` stay unchanged:

- single-frame matching versus subsequence-DTW;
- query length and query sampling stride;
- progress smoothing or allowing a new estimate to revise an old estimate downward;
- completion threshold and number of confirmations;
- `reference_clock`, oracle, or video-derived progress.

The training contract would change only if progress semantics/normalization, stage numbering,
state slots, action horizon, or the model input modalities change.

## 2. What `query_length=8` means

It does not limit the reference video to eight frames. The complete current-stage reference is
kept at stride-8 nodes. Only the rolling rollout query keeps the eight most recent sampled RGB
descriptors.

At 20 Hz with `query_stride=8`, one query node is collected every 0.4 seconds. Eight nodes span
2.8 seconds from the first to the last sample:

```text
t-2.8, t-2.4, t-2.0, t-1.6, t-1.2, t-0.8, t-0.4, t
```

This short visual history provides motion order that a single image cannot provide. Eight is the
default because the previous held-out GT-video DTW experiment validated exactly this protocol;
it is not known to be optimal for policy rollout. Keep it fixed during training. After training,
compare at least query lengths `4, 8, 12, 16` with the same checkpoint.

## 3. Current implemented inference behavior

The current `reference_dtw` implementation uses:

- `agentview_left` RGB32 descriptors;
- stride-8 reference and rollout query nodes;
- rolling query length 8, with a shorter partial query during startup;
- zero motion, stay, and jump penalties;
- no hard maximum reference span inside a query;
- a non-decreasing endpoint across policy calls;
- stage advance after `progress >= 0.9` on two policy calls;
- a stage index that can hold or advance by one, but never regress or skip.

These values are a runnable inference configuration, not part of the training contract.

## 4. Recommended rule to evaluate after training

The first post-training alternative should make each rolling DTW query independently able to
correct an earlier progress estimate while keeping the stage index monotonic:

```text
collect a full 8-node query
        |
run DTW for the current stage
        |
progress may increase or decrease when new visual evidence changes the match
        |
first reliable progress >= 0.9:
    keep the current stage and execute one 16-step finishing chunk
        |
second consecutive reliable progress >= 0.9:
    advance exactly one stage, reset progress to 0, clear the old query history
```

If the second estimate falls below 0.9, reset the completion streak and do not switch. The stage
index never moves backward even though within-stage progress may be revised.

The existing hard non-decreasing endpoint makes a single false high-progress match permanent, so
two confirmations then provide delay rather than genuine error rejection. Compare that current
behavior against the independently re-estimated rule before selecting the final controller.

DTW mean cost and confidence margin should be logged from the start, but no confidence threshold
should be chosen until its validation distribution is measured. Do not tune this decision on the
locked test split.

## 5. Post-training evaluation order

Use one fixed clean oracle-conditioned checkpoint:

1. oracle `(stage, progress)` to measure the conditioned-policy upper bound;
2. `reference_clock` to validate policy/environment plumbing;
3. the current `reference_dtw` settings;
4. query-length and progress-revision ablations;
5. transition timing, premature-switch rate, late-switch rate, node/progress error, and success;
6. only after a clean positive signal, consider training-time `±1` reference-node noise.

This order keeps localizer/controller differences attributable to inference rather than to
different training runs.
