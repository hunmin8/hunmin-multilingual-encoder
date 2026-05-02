# Hunmin 30M Simple G Runs

## Why

The best 13M multilingual result came from the simple recipe:

- `13.8M`
- `7lang`
- `EN-hub text-only`
- `40K steps`
- no Hunmin/Jamo auxiliary complexity

The new 30M runs test whether the same simple recipe scales better when the
model is larger.

## Active Runs

### g1

- run: `hunmin-small-30m-7lang-enhub2m-textonly-simple-40k-g1-run1`
- data: `output/enhub_sentence_v2_text_only_6lang_2m_clean`
- steps: `40K`
- seed: `20260521`
- purpose: direct 30M version of the stable 2M clean setup

### g2

- run: `hunmin-small-30m-7lang-enhub3m-textonly-simple-40k-g2-run1`
- data: `output/enhub_sentence_v2_text_only_6lang_3m_clean`
- steps: `40K`
- seed: `20260522`
- purpose: check whether 3M data improves over 2M for 30M

### g3

- run: `hunmin-small-30m-7lang-enhub3m-clean-g3-long20k`
- status: left running, not touched
- purpose: keep the best existing 30M g run as continuity baseline

### g4

- run: `hunmin-small-30m-7lang-enhub3m-textonly-simple-40k-g4-seed2-run1`
- data: `output/enhub_sentence_v2_text_only_6lang_3m_clean`
- steps: `40K`
- seed: `20260524`
- purpose: seed comparison against g2

## Success Criteria

Compare against:

- `hunmin-lite-13m-7lang-enhub-textonly-simple-40k-run2`
- current 30M g3 best

The 30M simple line is worth keeping if:

- R@1 clearly exceeds `0.60`
- R@5 rises with R@1, not just candidate recall
- g2/g4 seed variance is not too large

If all runs stay near `0.58`, the 13M simple recipe remains the better lite
baseline and 30M needs data cleanup rather than more architectural changes.
