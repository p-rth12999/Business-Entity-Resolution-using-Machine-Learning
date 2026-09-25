# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** [Date]

---

## 1. Executive Summary
We treat the task as cross-source business entity resolution: candidate pairs are generated with high-recall blocking and then scored by a precision-oriented pair classifier. The pipeline combines conservative text normalization, complementary name/address similarities, country and address-component evidence, and a validation threshold optimized for macro $F_{0.5}$.

---

## 2. Methodology

### 2.1 Problem Analysis
The data contains noisy business names and addresses across three sources. Important variations include punctuation and case differences, legal suffixes, abbreviations, typos, word-order changes, transliteration, missing address components, and landmark-based addresses. The test set also contains France although training contains US and India, so country must be treated as an open-set string field rather than a fixed categorical vocabulary. Since evaluation is macro $F_{0.5}$, false merges and incorrect predictions for singleton entities must be avoided.

### 2.2 Solution Strategy
The solution has four stages: shared normalization, multi-pass candidate generation, pairwise feature extraction, and binary classification followed by thresholding. Candidate generation is deliberately high recall; the final classifier makes the conservative merge decision. Validation splits are made by Source 1 entity to prevent leakage between related pairs.

**Approach Type:** Hybrid blocking + binary classifier  
**Core Innovation:** Union several inexpensive exact/partial blocks with approximate similarity retrieval, then calibrate a precision-oriented threshold using macro $F_{0.5}$ rather than accuracy.

---

## 3. Candidate Generation (Blocking)
We generate the union of several blocking passes and deduplicate the resulting Source 1-to-Source 2/3 pairs. Blocks use country plus name tokens, postal code plus a loose name key, house number plus address tokens, and approximate nearest neighbors over character TF-IDF representations. Embedding retrieval with `all-MiniLM-L6-v2` can be added for records that do not enter an exact block; it is an optional fallback rather than the only retrieval mechanism.

- **Blocking keys used:** Normalized country/name tokens, postal code, house number/address tokens, and character TF-IDF nearest neighbors.
- **Candidate pairs generated:** [fill from the final candidate-pair file]
- **How you ensured true matches were not lost:** Measure candidate recall against held-out ground truth and retain the union of blocks. The candidate file records the exact final set passed to the matching model.

---

## 4. Matching Model

**Features used:**
- Name features: normalized exact match, token Jaccard, character n-gram overlap, normalized Levenshtein/Jaro-Winkler similarity, and token-order-insensitive similarity.
- Address features: token overlap, edit similarity, postal-code match, house-number match, and city/state component matches.
- Other: country equality, missing-field flags, shared numeric-token count, string lengths, source-pair indicator, and blocking-rule indicators.

**Model type:** [fill with the selected binary classifier: XGBoost / LightGBM / CatBoost / other]  
**Threshold selection method:** Select the threshold on a Source-1-held-out validation split to maximize macro $F_{0.5}$. Use stricter handling for contradictory countries and weak name-only matches.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** [your best validation score]
- **Common false positives (wrong merges):** [brief description]
- **Common false negatives (missed matches):** [brief description]

---

## 6. Conclusion
The final system separates high-recall retrieval from conservative pair classification, which is appropriate for a precision-heavy metric and allows one-to-many matches. The main remaining work is empirical threshold and block tuning using held-out Source 1 entities, followed by validation of both required TSV outputs.

---

## Appendix

### A. Code Artefacts
*Your complete, runnable code ships in the submission zip under
`code/business_entity_resolution/` (all source in `src/`, with a `README.md` and
`requirements.txt`). Summarise its structure and the entry point(s) to reproduce
`output/matching_results.tsv` and `output/candidate_pairs.tsv` here.*

### B. Additional Results
*Include any additional charts, graphs, or detailed results.*

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
