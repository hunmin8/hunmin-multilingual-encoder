# Hunmin Multilingual Encoder

BGE 대체를 목표로 하는 Hunmin 다국어 임베딩 인코더 실험 레포입니다.

이 레포는 검색 레이어가 아니라 **인코더 학습/평가**만 다룹니다.

```text
input text -> embedding vector
```

## 현재 방향

단순한 12M 스타일 encoder 구조에서 출발해, 제대로 된 다국어 문장쌍 데이터로 키웁니다.

현재 메인 계열:

```text
hunmin-main-80m-7lang-enhub5m-direct1m-textonly-h100-run1
```

언어:

```text
en, ko, ja, zh, fr, de, es
```

## 현재 H100 상태

2026-05-02 확인:

```text
step 30,000
cross R@1 mean: 0.8035
R@5 mean:       0.8687
```

이 수치는 내부 validation 기준입니다. 외부 벤치마크와 별도로 검증해야 합니다.

## 레포 분리 원칙

- `hunmin`: 전사기 / UHPS / IPA
- `hunmin-ko-encoder`: 한국어 단독 인코더
- `hunmin-multilingual-encoder`: 다국어 인코더
- `hunmin-search`: 검색 API / FAISS / rerank / UI

## 실행 예시

H100 80M:

```bash
python scripts/train_hunmin_m12_canonical.py \
  --config configs/hunmin_main_80m_7lang_enhub5m_direct1m_textonly_h100_run1.json \
  --train output/enhub_sentence_v1_text_only_6lang_5m_plus_direct_v1/train.jsonl \
  --val output/enhub_sentence_v1_text_only_6lang_5m_plus_direct_v1/val.jsonl \
  --output-dir models/hunmin-main-80m-7lang-enhub5m-direct1m-textonly-h100-run1
```

외부 벤치마크:

```bash
python scripts/eval_external_benchmarks.py \
  --encoder hunmin \
  --checkpoint models/hunmin-main-80m-7lang-enhub5m-direct1m-textonly-h100-run1/best.pt \
  --device cuda
```

## 금지

- 검색 rerank로 모델 성능을 부풀리지 않는다.
- BGE를 production dependency로 쓰지 않는다.
- 전사기/scribe를 이 레포에서 수정하지 않는다.
- 모델 평가는 internal과 external을 분리한다.

## 다음 할 일

1. H100 80M 5M+direct run 완료
2. `queue_eval_multibench_after_80m_5m_direct_run1.sh`로 완료 후 자동 평가
3. checkpoint/data/config 재현 패키징
4. FLORES/KLUE STS/다국어 retrieval 평가
5. 155M 또는 더 큰 데이터로 확장
6. Hugging Face 모델 카드 작성
