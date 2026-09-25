# HANDOFF — Amazon ML Challenge: Business Entity Resolution

**From:** Coder 1 (foundation phase) **To:** Coder 2 (optimization phase)
**Status:** Full baseline pipeline runs end to end. Numbers below need to be filled in after running it on real data.

---

## 1. Pipeline status

| Step | Module | Status |
|---|---|---|
| 1 — Load & inspect | `io_utils.py` | Done |
| 2 — Normalize | `normalize.py` | Done — both name representations |
| 3 — Pairwise labels | `labels.py` | Done — hard negatives only |
| 4 — Blocking | `blocking.py` | Done — 5 rule-based blocks |
| 5 — Features | `features.py` | Done |
| 6 — Model | `model.py` | Done — LightGBM, entity-level split |
| 7 — Threshold tuning | `threshold.py` | **Not started — this is you** |
| 8 — Evaluation | `evaluate.py` | Done — macro F0.5, error examples |
| 9 — Singleton handling | (built into evaluate.py + predict.py) | Done |
| 10 — Test inference | `predict.py`, `main.py` | Done — untested against real test data yet |

---

## 2. Current numbers

*(Fill in after running `python main.py train` then `python main.py evaluate` on the actual dataset)*

- Blocking recall: `____`
- Validation macro F0.5 @ threshold=0.5 (default, untuned): `____`
- Threshold sweep (from `python main.py evaluate`):

| threshold | macro F0.5 |
|---|---|
| 0.30 | |
| 0.40 | |
| 0.50 | |
| 0.60 | |
| 0.70 | |
| 0.80 | |
| 0.90 | |

---

## 3. What's already built

- **Blocking:** exact normalized name, postal-code + name-token, house-number + address-token-overlap, country + name-prefix, TF-IDF character n-gram nearest neighbors — union of all five.
- **Features:** name (exact, stripped-exact, Jaccard, Levenshtein, Jaro-Winkler, TF-IDF cosine), address (exact, Jaccard, Levenshtein), postal/house-number/country match, missing-field flags, source + blocking-rule indicators, 3 interaction features.
- **Model:** LightGBM, single global threshold (not yet tuned — currently `config.DEFAULT_THRESHOLD = 0.5`), hard negatives only.
- **Validation:** entity-level split across the *full* ground-truth list, including entities with zero candidates — this was a real bug in an earlier version, now fixed in `model.split_entity_ids()`.

---

## 4. Concrete next steps

1. Run `python main.py train` and `python main.py evaluate` — fill in Section 2 above.
2. **Build `threshold.py`:** don't just eyeball the sweep — persist the chosen threshold (e.g. write it to a small json/config value) so `predict.py` uses the tuned number instead of the hardcoded default.
3. **If blocking recall is low:** check which block is weak — `evaluate.get_error_examples()` and the `blocking_rules` column will show which rule(s) fired (or didn't) for missed matches.
4. **If blocking recall is high but F0.5 is low:** it's a feature/model problem — pull false positives and false negatives via `evaluate.get_error_examples()` and look for patterns.
5. **Per-source threshold split (S2 vs S3):** only pursue this if validation shows a real, consistent score-distribution gap between the two — see decision #5 in the methodology doc.
6. **Error-analysis write-up:** feed `get_error_examples()` output into `Documentation_template.md` Section 5.
7. **Before packaging:** run `utils/validate_submission.py`, confirm `output/candidate_pairs.tsv` reflects the actual final scored set (it does, via `predict.py`), delete `test.ipynb` if it's still around.

---

## 5. Known limitations to watch

- The TF-IDF nearest-neighbor block does brute-force cosine similarity (sklearn) — fine at hackathon scale; may need batching if a source table turns out to be very large.
- Postal-code extraction uses a "longest digit group" heuristic — worth double-checking once French test addresses are actually in hand.
- `predict.py` currently hardcodes `config.DEFAULT_THRESHOLD` — this is the first thing that should change once `threshold.py` exists.

---

## 6. File map

All under `code/business_entity_resolution/src/`:
`config.py` · `io_utils.py` · `normalize.py` · `blocking.py` · `labels.py` · `features.py` · `model.py` · `evaluate.py` · `predict.py` · `main.py`

Run anything with:
```powershell
cd code\business_entity_resolution\src
python main.py train      # trains + saves model, prints validation F0.5
python main.py evaluate   # trains + threshold sweep, no save
python main.py predict    # loads saved model, scores test data, writes submission files
```