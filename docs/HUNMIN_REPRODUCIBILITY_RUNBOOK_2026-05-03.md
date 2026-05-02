# Hunmin Encoder Reproducibility Runbook

작성일: 2026-05-03

이 문서는 나중에 같은 모델을 다시 만들 수 있도록, 현재까지 중요한 인코더 실험의
코퍼스, 데이터 출처, 검증 조건, 언어쌍, 스텝수, 서버, 스크립트, 결과 위치를 정리한다.

## 0. 기준 폴더

| 위치 | 경로 |
|---|---|
| mac 로컬 노트 | `/Users/dragon/Documents/New project/hunmin_notes/` |
| d2 메인 보관 | `/home/dragon/hunmin_v1/` |
| H100 다국어 | `/root/hunmin_v1/` on `root@216.243.220.230:18861` |
| H100 한국어 | `/root/hunmin_v1/` on `root@216.243.220.226:13353` |
| g1-g4 | `/home/dragon/hunmin_v1/` |

## 1. 공통 학습 코드

대부분의 인코더 실험은 같은 학습 스크립트를 사용한다.

```text
scripts/train_hunmin_m12_canonical.py
```

공통 방식:

- Transformer encoder
- mean/pooled sentence embedding
- InfoNCE 계열 contrastive objective
- text pair 중심
- tokenizer: `CharTagTokenizer`
- BPE 학습 없음
- special tokens: `[EN]`, `[KO]`, `[JA]`, `[ZH]`, `[FR]`, `[DE]`, `[ES]`, `[HUNMIN]`, `[UHPS]`

중요:

```text
전사기/scribe는 production corpus generator 또는 별도 기능이고,
현재 다국어 인코더 메인 실험은 text-only simple 방식이 가장 강했다.
```

## 2. 데이터셋 계열

### 2.1 EN-hub sentence v1 text-only 6lang

위치:

```text
output/enhub_sentence_v1_text_only_6lang/
```

사용 모델:

```text
hunmin-lite-13m-7lang-enhub-textonly-simple-40k-run2
hunmin-lite-13m-7lang-enhub-textonly-simple-80k-run1
```

출처:

| source | pair | max rows |
|---|---|---:|
| `/home/dragon/hunmin/data/opus_en_ko.tsv` | EN-KO | 250K |
| `/home/dragon/hunmin/data/opus_en_ja.tsv` | EN-JA | 250K |
| `/home/dragon/hunmin/data/opus_en_zh.tsv` | EN-ZH | 250K |
| `/home/dragon/hunmin/data/opus_en_fr.tsv` | EN-FR | 250K |
| `/home/dragon/hunmin/data/opus_de_en.tsv` | DE-EN | 250K |
| `/home/dragon/hunmin/data/opus_en_es.tsv` | EN-ES | 250K |

규모:

| split | rows |
|---|---:|
| train | 1,200,000 |
| val | 120,000 |
| test | 120,000 |

언어별:

| lang | total |
|---|---:|
| ko | 240K |
| ja | 240K |
| zh | 240K |
| fr | 240K |
| de | 240K |
| es | 240K |

필터:

- `min_chars=4`
- `max_chars=240`
- `max_length_ratio=3.5`
- `max_latin_jaccard=0.88`
- `strict_script=true`
- `one_translation_per_en=true`
- `aux=none`
- `include_phonetic_cross=false`

판단:

```text
첫 강한 baseline. simple text-only EN-hub가 맞다는 증거.
```

### 2.2 EN-hub sentence v2 text-only 6lang 2M clean

위치:

```text
output/enhub_sentence_v2_text_only_6lang_2m_clean/
```

사용 모델:

```text
hunmin-main-80m-7lang-enhub2m-clean-h100-run1
hunmin-small-30m-7lang-enhub2m-textonly-simple-40k-g1-run1
```

출처:

| source | pair | max rows |
|---|---|---:|
| `/home/dragon/hunmin/data/opus_en_ko.tsv` | EN-KO | 1.2M |
| `/home/dragon/hunmin/data/opus_en_ja.tsv` | EN-JA | 1.2M |
| `/home/dragon/hunmin/data/opus_en_zh.tsv` | EN-ZH | 1.2M |
| `/home/dragon/hunmin/data/opus_en_fr.tsv` | EN-FR | 1.2M |
| `/home/dragon/hunmin/data/opus_de_en.tsv` | DE-EN | 1.2M |
| `/home/dragon/hunmin/data/opus_en_es.tsv` | EN-ES | 1.2M |

규모:

| split | rows |
|---|---:|
| train | 1,999,998 |
| val | 115,363 |
| test | 115,782 |

언어별 train:

| lang | train rows |
|---|---:|
| ko | 333,333 |
| ja | 333,333 |
| zh | 333,333 |
| fr | 333,333 |
| de | 333,333 |
| es | 333,333 |

필터:

- `min_chars=4`
- `max_chars=220`
- `max_length_ratio=3.3`
- `max_latin_jaccard=0.82`
- `strict_script=true`
- `one_translation_per_en=true`
- `aux=none`
- `include_phonetic_cross=false`

판단:

```text
현재 다국어 80M 최고 모델을 만든 데이터셋.
```

### 2.3 EN-hub sentence v2 text-only 6lang 3M clean

위치:

```text
output/enhub_sentence_v2_text_only_6lang_3m_clean/
```

사용 모델:

```text
hunmin-small-30m-7lang-enhub3m-textonly-simple-40k-g2-run1
hunmin-small-30m-7lang-enhub3m-textonly-simple-40k-g3-seed3-run1
hunmin-small-30m-7lang-enhub3m-textonly-simple-40k-g4-seed2-run1
hunmin-small-30m-7lang-enhub3m-clean-g3-long20k
```

출처:

| source | pair | max rows |
|---|---|---:|
| `/home/dragon/hunmin/data/opus_en_ko.tsv` | EN-KO | 1.6M |
| `/home/dragon/hunmin/data/opus_en_ja.tsv` | EN-JA | 1.6M |
| `/home/dragon/hunmin/data/opus_en_zh.tsv` | EN-ZH | 1.6M |
| `/home/dragon/hunmin/data/opus_en_fr.tsv` | EN-FR | 1.6M |
| `/home/dragon/hunmin/data/opus_de_en.tsv` | DE-EN | 1.6M |
| `/home/dragon/hunmin/data/opus_en_es.tsv` | EN-ES | 1.6M |

규모:

| split | rows |
|---|---:|
| train | 2,848,603 |
| val | 115,363 |
| test | 115,782 |

언어별 train:

| lang | train rows |
|---|---:|
| ko | 500,000 |
| ja | 500,000 |
| zh | 348,603 |
| fr | 500,000 |
| de | 500,000 |
| es | 500,000 |

필터:

- `min_chars=4`
- `max_chars=220`
- `max_length_ratio=3.3`
- `max_latin_jaccard=0.82`
- `strict_script=true`
- `one_translation_per_en=true`
- `aux=none`
- `include_phonetic_cross=false`

주의:

```text
ZH 원본 공급량이 부족해서 train이 348,603으로 낮다.
```

### 2.4 EN-hub 2M + direct10 clean v1

위치:

```text
output/enhub_sentence_v2_text_only_6lang_2m_plus_direct10_clean_v1/
```

사용 모델:

```text
hunmin-main-80m-7lang-enhub2m-direct10-clean-h100-run1
```

구성:

| component | rows |
|---|---:|
| base EN-hub 2M train | 1,999,998 |
| direct supplement train | 200,000 |
| final train | 2,199,998 |
| final val | 127,363 |
| test | 115,782 |

direct train quota:

| pair | rows |
|---|---:|
| JA-KO | 45,000 |
| KO-ZH | 45,000 |
| JA-ZH | 40,000 |
| DE-JA | 12,000 |
| FR-JA | 12,000 |
| ES-JA | 12,000 |
| DE-KO | 12,000 |
| FR-KO | 11,000 |
| ES-KO | 11,000 |

direct source:

```text
output/enhub_sentence_v1_text_only_6lang_5m_plus_direct_clean_v2/train.direct_clean.jsonl
```

실제 direct source 대부분:

```text
WikiMatrix direct pairs
```

중요한 주의:

```text
이 데이터셋은 최종 품질 데이터가 아니라 direct signal 효과 검증용이다.
KO-ZH에서 gold 오염이 확인되었으므로, production-quality direct는 별도 filtered v3가 필요하다.
```

## 3. 주요 모델 재현 정보

### 3.1 Lite 13M 7lang EN-hub text-only simple 40K

| 항목 | 값 |
|---|---|
| model | `hunmin-lite-13m-7lang-enhub-textonly-simple-40k-run2` |
| params | 약 13.8M |
| server | d2 |
| workdir | `/home/dragon/hunmin_v1` |
| dataset | `output/enhub_sentence_v1_text_only_6lang/` |
| config | `configs/hunmin_lite_13m_7lang_enhub_textonly_simple_40k_run2.json` |
| start script | `scripts/start_hunmin_lite_13m_7lang_enhub_textonly_simple_40k_run2.sh` |
| train script | `scripts/train_hunmin_m12_canonical.py` |
| train file | `output/enhub_sentence_v1_text_only_6lang/train.shuffled.jsonl` |
| val file | `output/enhub_sentence_v1_text_only_6lang/val.shuffled.jsonl` |
| output | `models/hunmin-lite-13m-7lang-enhub-textonly-simple-40k-run2/` |
| steps | 40,000 |
| batch | 128 |
| lr | 2e-4 |
| vocab | max 8,000 |
| max_len | 160 |
| seed | 20260501 |
| best step | 38,000 |
| best R@1 | 0.6725 |
| final R@1 | 0.6720 |
| elapsed | 약 3,116 sec / 52 min |

재현 명령:

```bash
cd /home/dragon/hunmin_v1
scripts/start_hunmin_lite_13m_7lang_enhub_textonly_simple_40k_run2.sh
```

### 3.2 Lite 13M 7lang EN-hub text-only simple 80K

| 항목 | 값 |
|---|---|
| model | `hunmin-lite-13m-7lang-enhub-textonly-simple-80k-run1` |
| params | 약 13.8M |
| server | d2 |
| dataset | `output/enhub_sentence_v1_text_only_6lang/` |
| config | `configs/hunmin_lite_13m_7lang_enhub_textonly_simple_80k_run1.json` |
| output | `models/hunmin-lite-13m-7lang-enhub-textonly-simple-80k-run1/` |
| steps | 80,000 |
| batch | 128 |
| lr | 2e-4 |
| best step | 74,000 |
| best R@1 | 0.7290 |
| final R@1 | 0.7283 |
| elapsed | 약 6,232 sec / 1h 44m |

판단:

```text
13M은 단순히 40K에서 끝낼 필요가 없었다. 80K까지 가면 크게 오른다.
```

### 3.3 Main 80M 7lang EN-hub 2M clean

| 항목 | 값 |
|---|---|
| model | `hunmin-main-80m-7lang-enhub2m-clean-h100-run1` |
| params | 약 80M |
| server | H100 multilingual |
| ssh | `root@216.243.220.230 -p 18861` |
| workdir | `/root/hunmin_v1` |
| dataset | `output/enhub_sentence_v2_text_only_6lang_2m_clean/` |
| config | `configs/hunmin_main_80m_7lang_enhub2m_clean_h100_run1.json` |
| start script | `scripts/start_hunmin_main_80m_7lang_enhub2m_clean_h100_run1.sh` |
| train script | `scripts/train_hunmin_m12_canonical.py` |
| train file | `output/enhub_sentence_v2_text_only_6lang_2m_clean/train.shuffled.jsonl` |
| val file | `output/enhub_sentence_v2_text_only_6lang_2m_clean/val.shuffled.jsonl` |
| output | `models/hunmin-main-80m-7lang-enhub2m-clean-h100-run1/` |
| steps | 60,000 |
| batch | 512 |
| lr | 1e-4 |
| vocab | max 12,000 |
| max_len | 160 |
| seed | 20260506 |
| best step | 56,000 |
| best R@1 | 0.8116 |
| final R@1 | 0.8113 |
| elapsed | 약 28,095 sec / 7h 48m |

재현 명령:

```bash
cd /root/hunmin_v1
scripts/start_hunmin_main_80m_7lang_enhub2m_clean_h100_run1.sh
```

판단:

```text
현재 다국어 main 최고 기준점.
```

### 3.4 Main 80M 7lang EN-hub 2M + direct10

| 항목 | 값 |
|---|---|
| model | `hunmin-main-80m-7lang-enhub2m-direct10-clean-h100-run1` |
| params | 약 80M |
| server | H100 multilingual |
| dataset | `output/enhub_sentence_v2_text_only_6lang_2m_plus_direct10_clean_v1/` |
| config | `configs/hunmin_main_80m_7lang_enhub2m_direct10_clean_h100_run1.json` |
| start script | `scripts/start_hunmin_main_80m_7lang_enhub2m_direct10_clean_h100_run1.sh` |
| queue script | `scripts/queue_direct10_after_enhub2m_clean.sh` |
| eval script | `scripts/eval_hunmin_main_80m_direct10_compare.sh` |
| train file | `output/enhub_sentence_v2_text_only_6lang_2m_plus_direct10_clean_v1/train.shuffled.jsonl` |
| val file | `output/enhub_sentence_v2_text_only_6lang_2m_plus_direct10_clean_v1/val.shuffled.jsonl` |
| output | `models/hunmin-main-80m-7lang-enhub2m-direct10-clean-h100-run1/` |
| steps | 60,000 |
| batch | 512 |
| lr | 1e-4 |
| seed | 20260507 |
| current step | 28,800 at last metadata pull |
| current best step | 28,000 |
| current best R@1 | 0.8109 |
| status | running |

실험 목적:

```text
오염이 조금 있는 raw direct라도 약한 직접 언어쌍에 도움이 되는지 확인.
```

판단 기준:

- 전체 EN-hub R@1이 크게 깨지면 실패
- direct pair, 특히 `KO-ZH`, `DE-KO`, `FR-KO`, `ES-KO`가 오르면 성공 신호

### 3.5 Small 30M 7lang EN-hub 2M simple 40K g1

| 항목 | 값 |
|---|---|
| model | `hunmin-small-30m-7lang-enhub2m-textonly-simple-40k-g1-run1` |
| params | 약 30M |
| server | g1 |
| workdir | `/home/dragon/hunmin_v1` |
| dataset | `output/enhub_sentence_v2_text_only_6lang_2m_clean/` |
| config | `configs/hunmin_small_30m_7lang_enhub2m_textonly_simple_40k_g1_run1.json` |
| start script | `scripts/start_hunmin_small_30m_7lang_enhub2m_textonly_simple_40k_g1_run1.sh` |
| steps | 40,000 |
| batch | 128 |
| lr | 2e-4 |
| seed | 20260521 |
| last observed step | 18,000 |
| best R@1 observed | 0.6330 |
| status | running |

### 3.6 Small 30M 7lang EN-hub 3M simple 40K g2

| 항목 | 값 |
|---|---|
| model | `hunmin-small-30m-7lang-enhub3m-textonly-simple-40k-g2-run1` |
| params | 약 30M |
| server | g2 |
| dataset | `output/enhub_sentence_v2_text_only_6lang_3m_clean/` |
| config | `configs/hunmin_small_30m_7lang_enhub3m_textonly_simple_40k_g2_run1.json` |
| start script | `scripts/start_hunmin_small_30m_7lang_enhub3m_textonly_simple_40k_g2_run1.sh` |
| steps | 40,000 |
| batch | 128 |
| lr | 2e-4 |
| seed | 20260522 |
| last observed step | 18,000 |
| best R@1 observed | 0.6454 |
| status | running |

### 3.7 Small 30M 7lang EN-hub 3M simple 40K g4 seed2

| 항목 | 값 |
|---|---|
| model | `hunmin-small-30m-7lang-enhub3m-textonly-simple-40k-g4-seed2-run1` |
| params | 약 30M |
| server | g4 |
| dataset | `output/enhub_sentence_v2_text_only_6lang_3m_clean/` |
| config | `configs/hunmin_small_30m_7lang_enhub3m_textonly_simple_40k_g4_seed2_run1.json` |
| start script | `scripts/start_hunmin_small_30m_7lang_enhub3m_textonly_simple_40k_g4_seed2_run1.sh` |
| steps | 40,000 |
| batch | 128 |
| lr | 2e-4 |
| seed | 20260524 |
| last observed step | 18,000 |
| best R@1 observed | 0.6465 |
| status | running |

### 3.8 Small 30M 7lang EN-hub 3M clean long20K g3

| 항목 | 값 |
|---|---|
| model | `hunmin-small-30m-7lang-enhub3m-clean-g3-long20k` |
| params | 약 30M |
| server | g3 |
| dataset | `output/enhub_sentence_v2_text_only_6lang_3m_clean/` |
| steps | 20,000 |
| best/final R@1 | 0.6353 |
| status | completed |

판단:

```text
20K로도 13M 40K를 넘었지만, 40K simple seed runs가 더 좋게 오르는 중.
```

### 3.9 Small 30M 7lang EN-hub 3M simple 40K g3 seed3

| 항목 | 값 |
|---|---|
| model | `hunmin-small-30m-7lang-enhub3m-textonly-simple-40k-g3-seed3-run1` |
| params | 약 30M |
| server | g3 |
| dataset | `output/enhub_sentence_v2_text_only_6lang_3m_clean/` |
| config | `configs/hunmin_small_30m_7lang_enhub3m_textonly_simple_40k_g3_seed3_run1.json` |
| start script | `scripts/start_hunmin_small_30m_7lang_enhub3m_textonly_simple_40k_g3_seed3_run1.sh` |
| steps | 40,000 |
| batch | 128 |
| lr | 2e-4 |
| seed | 20260523 |
| status | running |

## 4. 한국어 단독 모델

### 4.1 KO 80M corpus v6 balanced

| 항목 | 값 |
|---|---|
| model | `hunmin-ko-80m-1lang-corpus-v6-balanced-h100-run1` |
| params | 약 80M |
| server | H100 Korean |
| ssh | `root@216.243.220.226 -p 13353` |
| dataset | `output/ko_mono_corpus_v6_h100_balanced/` |
| config | `configs/hunmin_ko_80m_1lang_corpus_v6_balanced_h100_run1.json` |
| output | `models/hunmin-ko-80m-1lang-corpus-v6-balanced-h100-run1/` |
| steps | 80,000 |
| batch | 512 |
| lr | 1e-4 |
| vocab | max 16,000 |
| max_len | 160 |
| seed | 20260502 |
| best/final R@1 | 0.7852 |
| elapsed | 약 38,208 sec / 10h 37m |

데이터 구성:

| split | total rows |
|---|---:|
| train | 412,754 |
| val | 19,000 |
| test | 18,881 |

train record kinds:

| kind | rows |
|---|---:|
| semantic_pair | 162,523 |
| query_doc_synthetic | 110,231 |
| entity_query_doc | 90,000 |
| persona_query_doc | 50,000 |

hard negatives:

```text
80,000
```

### 4.2 KO 80M corpus v7 mixed-English

| 항목 | 값 |
|---|---|
| model | `hunmin-ko-80m-1lang-corpus-v7-mixed-english-h100-run1` |
| params | 약 80M |
| server | H100 Korean |
| dataset | `output/ko_mono_corpus_v7_mixed_english_ready/` |
| config | `configs/hunmin_ko_80m_1lang_corpus_v7_mixed_english_h100_run1.json` |
| output | `models/hunmin-ko-80m-1lang-corpus-v7-mixed-english-h100-run1/` |
| steps | 60,000 |
| batch | 512 |
| lr | 1e-4 |
| seed | 20260503 |
| current observed step | 43,350 |
| current best step | 42,000 |
| current best R@1 | 0.7428 |
| status | running |

데이터 구성:

| split | base rows | mixed-English rows | total |
|---|---:|---:|---:|
| train | 412,754 | 25,000 | 437,754 |
| val | 19,000 | 1,000 | 20,000 |
| test | 18,881 | 1,000 | 19,881 |

train record kinds:

| kind | rows |
|---|---:|
| semantic_pair | 162,523 |
| query_doc_synthetic | 110,231 |
| entity_query_doc | 90,000 |
| persona_query_doc | 50,000 |
| mixed_english_query_doc | 25,000 |

hard negatives:

| type | rows |
|---|---:|
| base hard negatives | 80,000 |
| mixed-English hard negatives | 12,500 |
| total | 92,500 |

목적:

```text
한국어 문서 안의 GPT, NVIDIA, iPhone, API 같은 영어 혼입 검색을 개선.
```

## 5. 평가/검증 스크립트

### 5.1 일반 학습 중 eval

학습 스크립트 내부 eval:

```text
text_a -> text_b
text_b -> text_a
text_a -> meaning
text_b -> meaning
score_recall_at_1_mean
```

주의:

```text
score_recall_at_1_mean은 모델별 내부 validation 기준이다.
외부 벤치나 다른 데이터셋 R@1과 직접 비교하면 안 된다.
```

### 5.2 다국어 multi-benchmark eval

스크립트:

```text
scripts/eval_m12_multibench.py
```

direct10 비교 스크립트:

```text
scripts/eval_hunmin_main_80m_direct10_compare.sh
```

direct10 baseline 결과 위치:

```text
models/hunmin-main-80m-7lang-enhub2m-clean-h100-run1/eval_direct10_compare_baseline_enhub2m_clean/
```

출력:

```text
metrics.json
metrics.md
summary.tsv
```

baseline direct pair summary:

| pair | R@1 | R@5 | MRR |
|---|---:|---:|---:|
| JA-KO | 0.8202 | 0.9102 | 0.8606 |
| JA-ZH | 0.8066 | 0.8973 | 0.8475 |
| KO-ZH | 0.3858 | 0.5570 | 0.4708 |
| DE-KO | 0.5369 | 0.7369 | 0.6280 |
| FR-KO | 0.6100 | 0.7846 | 0.6907 |
| ES-KO | 0.6554 | 0.8092 | 0.7267 |

해석:

```text
KO-ZH가 낮은 것은 모델만의 문제가 아니라 WikiMatrix KO-ZH gold 오염이 큼.
```

## 6. 서버별 현재 역할

| server | role |
|---|---|
| H100 multilingual | 80M main/directed experiments |
| H100 Korean | 80M Korean-only experiments |
| g1 | 30M 2M simple |
| g2 | 30M 3M simple seed1 |
| g3 | 30M 3M simple seed3 / completed old g3 baseline |
| g4 | 30M 3M simple seed2 |
| d2 | archive, baseline, scripts, docs |
| MacBook | orchestration and notes |

## 7. 재현 체크리스트

새 실험을 시작할 때 반드시 기록:

- model name에 크기/언어수/데이터/step 포함
- config path
- start script path
- train file
- val file
- dataset stats path
- source list
- split row counts
- language/pair counts
- filter 조건
- server
- start time
- end time
- best step
- best metric
- model output path
- eval output path
- 실패/성공 판단

## 8. 현재 고정 전략

```text
1. simple text-only EN-hub를 main baseline으로 둔다.
2. 13M/30M/80M을 같은 recipe로 비교한다.
3. direct pair는 효과 검증 후 quality filtering을 거친다.
4. KO mini-hub는 filtered direct가 준비된 뒤 넣는다.
5. STS-only fine-tuning은 retrieval을 망가뜨릴 수 있으므로 분리한다.
```

## 9. 추가로 반드시 보존해야 할 것

아래 항목은 사용자가 명시하지 않았지만, 모델 재현과 제품화 판단에 반드시 필요하다.

### 9.1 코드 버전 / git 상태

모델이 좋아도 어떤 코드로 만들었는지 모르면 재현할 수 없다.

각 run마다 기록:

```bash
cd /home/dragon/hunmin_v1
git rev-parse HEAD
git status --short
git diff --stat
```

보존 파일 권장:

```text
models/<RUN_NAME>/repro/git_commit.txt
models/<RUN_NAME>/repro/git_status.txt
models/<RUN_NAME>/repro/git_diff_stat.txt
```

주의:

```text
dirty worktree에서 만든 모델은 반드시 diff를 같이 저장한다.
```

### 9.2 실행 환경

같은 config라도 CUDA/PyTorch/Python 버전 차이로 속도와 결과가 달라질 수 있다.

각 run마다 기록:

```bash
python --version
python -c "import torch; print(torch.__version__, torch.version.cuda)"
nvidia-smi
pip freeze | sort
uname -a
df -h .
```

보존 파일 권장:

```text
models/<RUN_NAME>/repro/env.txt
models/<RUN_NAME>/repro/pip_freeze.txt
models/<RUN_NAME>/repro/nvidia_smi.txt
models/<RUN_NAME>/repro/disk.txt
```

### 9.3 하드웨어 정보

성능/비용 판단에 필요하다.

기록:

| 항목 | 예 |
|---|---|
| GPU | H100 / RTX 계열 / 내장 없음 |
| VRAM | 80GB / 24GB 등 |
| CPU | core count |
| RAM | system memory |
| disk | NVMe/HDD, free space |
| server | h100 multilingual / h100 ko / g1-g4 / d2 |

특히 기록할 것:

```text
tokens/sec 또는 steps/sec
총 elapsed time
GPU utilization
max VRAM
온도
```

### 9.4 데이터 무결성 / checksum

나중에 같은 이름의 파일이 바뀌면 재현이 깨진다.

각 데이터셋마다 저장:

```bash
sha256sum train.shuffled.jsonl val.shuffled.jsonl test.jsonl stats.json
wc -l train.shuffled.jsonl val.shuffled.jsonl test.jsonl
du -h train.shuffled.jsonl val.shuffled.jsonl test.jsonl
```

보존 파일:

```text
output/<DATASET>/MANIFEST.sha256
output/<DATASET>/MANIFEST.lines
output/<DATASET>/MANIFEST.size
```

모델 artifact도 동일하게:

```bash
sha256sum best.pt last.pt vocab.json metadata.json config.json
```

### 9.5 tokenizer / vocab 보존

인코더 재현에서 가장 자주 놓치는 부분이다.

반드시 보존:

```text
models/<RUN_NAME>/vocab.json
models/<RUN_NAME>/metadata.json
configs/<CONFIG>.json
```

기록할 것:

| 항목 | 필요 이유 |
|---|---|
| vocab size | 같은 모델 구조라도 vocab이 다르면 재현 불가 |
| special tokens | `[KO]`, `[EN]` 등 태그 순서 중요 |
| max vocab | `8000`, `12000`, `16000` |
| vocab build lines | `vocab_max_lines` |
| max_len | truncation 영향 |

주의:

```text
같은 config라도 train file 순서가 달라지면 vocab이 달라질 수 있다.
best.pt만 보관하지 말고 vocab.json을 반드시 같이 보관한다.
```

### 9.6 평가셋 고정

R@1 숫자는 평가셋이 달라지면 의미가 없다.

각 결과는 반드시 아래와 같이 표기한다.

```text
metric: R@1
eval file: output/.../val.shuffled.jsonl
eval rows: 12000
sampling seed: ...
direction: bidirectional mean / text_a->text_b / text_b->text_a
```

금지:

```text
서로 다른 eval set의 R@1을 같은 표에서 직접 비교하지 않기.
```

예:

```text
13M v1 eval R@1 0.729
80M hard 12K R@1 0.785
```

이 둘은 평가셋이 다르면 직접 우열 비교가 아니다.

### 9.7 외부 벤치마크 분리

내부 retrieval R@1과 외부 STS/MTEB류는 다른 지표다.

분리해서 기록:

| 평가 종류 | 목적 |
|---|---|
| internal pair retrieval | 학습쌍/유사 구조 검색 성능 |
| direct pair retrieval | 비영어 직접쌍 연결 |
| KO query-doc | 한국어 검색 실사용 |
| STS Spearman | 문장 유사도 |
| NLI/paraphrase | 의미 추론/문장 관계 |
| domain retrieval | 특정 제품/회사/문서 검색 |

판단 원칙:

```text
검색 모델은 retrieval이 우선이다.
STS가 올라가도 retrieval이 떨어지면 검색용 모델로는 실패다.
```

### 9.8 데이터 오염 리포트

특히 WikiMatrix/direct pair는 반드시 오염 검사를 남긴다.

각 direct dataset마다 저장:

```text
pair_quality_report.json
pair_quality_report.md
bad_examples.jsonl
good_examples.jsonl
```

검사 항목:

| 항목 | 설명 |
|---|---|
| model cosine | baseline encoder 기준 gold pair cosine |
| rank | gold target rank |
| mutual top-k | 양방향으로 서로 top-k 안에 드는지 |
| length ratio | 길이 비율 이상치 |
| script check | 언어 태그와 실제 문자 일치 |
| number/entity overlap | 숫자/고유명사 보존 여부 |
| duplicate source/target | 중복 번역 문제 |

판단:

```text
raw WikiMatrix는 학습 재료가 아니라 후보 재료다.
quality filter를 통과한 것만 production direct corpus로 승격한다.
```

### 9.9 실패 실험 보존 방식

실패한 실험도 버리면 같은 실수를 반복한다.

구조:

```text
configs/_archive_failed/YYYYMMDD_<reason>/
models/_archive_failed/YYYYMMDD_<reason>/
output/logs/_archive_failed/YYYYMMDD_<reason>/
```

각 실패 실험에 남길 것:

```text
WHY_FAILED.md
config
log tail
best metric
failure reason
reuse/no-reuse decision
```

예:

```text
STS fine-tune:
STS Spearman은 상승했지만 retrieval R@1이 급락.
검색용 base로 사용하지 않음.
```

### 9.10 artifact packaging 규칙

좋은 모델은 반드시 묶어서 보관한다.

권장 패키지:

```text
artifacts/<RUN_NAME>_<YYYYMMDD>/
  best.pt
  last.pt
  vocab.json
  metadata.json
  config.json
  train.log
  eval_summary.tsv
  dataset_stats.json
  MANIFEST.sha256
  REPRO.md
```

압축:

```bash
tar -czf artifacts/<RUN_NAME>_<YYYYMMDD>.tar.gz artifacts/<RUN_NAME>_<YYYYMMDD>/
sha256sum artifacts/<RUN_NAME>_<YYYYMMDD>.tar.gz > artifacts/<RUN_NAME>_<YYYYMMDD>.tar.gz.sha256
```

### 9.11 백업 정책

중요한 것은 최소 3곳에 둔다.

| 위치 | 역할 |
|---|---|
| H100 | 현재 학습 |
| d2 | 1차 영구 보관 |
| s1 | 2차 백업 |
| MacBook | 문서/작은 config/scripts |

보관 우선순위:

1. `best.pt`
2. `vocab.json`
3. `metadata.json`
4. config
5. start/eval script
6. dataset stats
7. eval summary
8. runbook

삭제 가능:

```text
중간 checkpoint는 best/last 확보 후 선택적으로 삭제 가능.
대용량 raw dataset은 stats/manifest/source recipe가 있으면 재생성 가능.
```

삭제 금지:

```text
best.pt + vocab.json + config + dataset stats + eval summary
```

### 9.12 비용 / 시간 기록

실험 설계에 필요하다.

각 run마다 기록:

| 항목 | 예 |
|---|---|
| server hourly cost | RunPod/H100 비용 |
| start time | ISO timestamp |
| end time | ISO timestamp |
| total hours | elapsed |
| best step time | best가 나온 시점 |
| final step time | 종료 시점 |
| cost to best | best까지 비용 |
| cost to final | 전체 비용 |

중요:

```text
best가 56K에서 나오고 60K final과 차이가 작으면, 다음 run은 56K 근처 early stop 후보.
```

### 9.13 모델 이름 규칙

모델명에는 최소한 아래가 들어가야 한다.

```text
hunmin-<tier>-<size>-<langcount>lang-<data>-<recipe>-<steps>-<server>-<run>
```

예:

```text
hunmin-main-80m-7lang-enhub2m-clean-h100-run1
hunmin-small-30m-7lang-enhub3m-textonly-simple-40k-g2-run1
hunmin-ko-80m-1lang-corpus-v7-mixed-english-h100-run1
```

금지:

```text
best_model.pt
new_run
test2
final_final
```

### 9.14 모델 카드 초안 정보

나중에 Hugging Face에 올릴 때 필요한 정보도 지금부터 쌓는다.

모델 카드 필수 항목:

```text
model size
languages
training data summary
intended use
not intended use
metrics
known weaknesses
license/data provenance
example usage
embedding dimension
max sequence length
tokenizer notes
```

특히 known weaknesses:

```text
raw WikiMatrix direct pair contamination
low-resource direct pair instability
STS/retrieval tradeoff
```

### 9.15 제품 관점의 분리

인코더와 검색 레이어는 다른 산출물이다.

| 계층 | 산출물 | 평가 |
|---|---|---|
| encoder | embedding model | retrieval/STS/NLI |
| search layer | FAISS/cluster/rerank/taxonomy | user query success, latency |
| corpus | pairs/data recipe | quality report, coverage |
| scribe | transcriber | gold transcription accuracy |

문서에서도 항상 분리해서 적는다.

```text
인코더가 약한 것인지,
검색 레이어가 약한 것인지,
데이터셋이 더러운 것인지,
평가셋이 잘못된 것인지
```

이 네 가지를 섞지 않는다.
