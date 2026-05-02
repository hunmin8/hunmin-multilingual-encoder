# Hunmin 80M Direct10 Verification Plan

## Purpose

Current best multilingual run is the simple EN-hub 2M clean recipe. The next test
adds a small direct-pair bridge, not a new architecture.

Run:

- baseline: `hunmin-main-80m-7lang-enhub2m-clean-h100-run1`
- next: `hunmin-main-80m-7lang-enhub2m-direct10-clean-h100-run1`

## Dataset

Prepared on H100 multilingual:

`/root/hunmin_v1/output/enhub_sentence_v2_text_only_6lang_2m_plus_direct10_clean_v1/`

Files:

- `train.shuffled.jsonl`: 2M EN-hub clean + 200K direct clean
- `val.shuffled.jsonl`: base val + 12K direct clean
- `train.direct10.jsonl`: direct supplement only
- `val.direct10.jsonl`: direct validation only
- `stats.json`: quotas and evaluation contract

Direct train quotas:

- `ja-ko`: 45K
- `ko-zh`: 45K
- `ja-zh`: 40K
- `de-ja`, `fr-ja`, `es-ja`, `de-ko`: 12K each
- `fr-ko`, `es-ko`: 11K each

## What Must Improve

The direct10 run is successful only if these improve against the 2M clean
checkpoint on the same files:

- `direct10_val` mean R@1
- `ja-ko` direct R@1
- `ko-zh` direct R@1
- `ja-zh` direct R@1

## What Must Not Break

The run is a failure if the original EN-hub validation collapses:

- `base_enhub_val` mean R@1 regression greater than `0.03`
- `base_enhub_val` mean R@5 regression greater than `0.02`

## Evaluation Script

Remote script:

`/root/hunmin_v1/scripts/eval_hunmin_main_80m_direct10_compare.sh`

It evaluates:

- `base_enhub_val`
- `direct10_val`
- `mixed_direct10_val`

It writes:

- `metrics.json`
- `metrics.md`
- `summary.tsv`

## Execution Order

1. Let current 2M clean run finish.
2. Evaluate 2M clean `best.pt` with `eval_hunmin_main_80m_direct10_compare.sh`.
3. Start direct10 run.
4. Evaluate direct10 `best.pt` with the same script.
5. Compare `summary.tsv` files.

## Expected Benefit

If the hypothesis is right, direct non-English retrieval improves while the
simple EN-hub strength is preserved. This specifically targets the weakness
where EN-centered training makes EN-KO/EN-JA/EN-ZH strong but direct JA-KO,
KO-ZH, and JA-ZH less stable.
