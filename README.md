# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Ctrl Win\
**Team Members:** Parth Kulkarni , Sanika Gadhave , Sekh Phaiyaz Alam\
**Submission Date:** 27 September 2026

------------------------------------------------------------------------

## 1. Executive Summary

We treat the task as cross-source business entity resolution: candidate
pairs are generated with high-recall blocking and then scored by a
precision-oriented pair classifier. The pipeline combines conservative
text normalization, complementary name/address similarities, country and
address-component evidence, and a validation threshold optimized for
macro $F_{0.5}$.

The final working pipeline successfully completed normalization,
blocking, feature extraction, model training, threshold selection, and
test prediction. The validation macro $F_{0.5}$ score obtained was
**0.7695**. The selected decision threshold was **0.57**.

For the final test prediction, the pipeline generated **32,501,248
candidate pairs** and produced matching results for **1,732,544 Source 1
entities**, with **1,438,006 matched entities** and **294,538 predicted
singletons**.

------------------------------------------------------------------------

## 2. Methodology

### 2.1 Problem Analysis

The data contains noisy business names and addresses across three
sources. Important variations include punctuation and case differences,
legal suffixes, abbreviations, typos, word-order changes,
transliteration, missing address components, and landmark-based
addresses. The test set also contains France although training contains
US and India, so country must be treated as an open-set string field
rather than a fixed categorical vocabulary. Since evaluation is macro
$F_{0.5}$, false merges and incorrect predictions for singleton entities
must be avoided.

### 2.2 Solution Strategy

The solution has four stages: shared normalization, multi-pass candidate
generation, pairwise feature extraction, and binary classification
followed by thresholding. Candidate generation is deliberately high
recall; the final classifier makes the conservative merge decision.
Validation splits are made by Source 1 entity to prevent leakage between
related pairs.

**Approach Type:** Hybrid blocking + binary classifier

**Core Innovation:** Union of four inexpensive, scalable exact/partial-match
blocking rules. A fifth, approximate-similarity fuzzy block (TF-IDF nearest
neighbors, and separately a scalable sorted-neighborhood variant) was
designed and implemented but intentionally left disabled in the final
submitted run due to data-scale constraints (see Section 3). The pipeline
then calibrates a precision-oriented threshold using macro $F_{0.5}$ rather
than accuracy.

The implementation was designed to process the large candidate volume in
chunks during feature extraction, reducing memory pressure while
allowing the complete candidate set to be processed successfully.

------------------------------------------------------------------------

## 3. Candidate Generation (Blocking)

We generate the union of four blocking passes and deduplicate the
resulting Source 1-to-Source 2/3 pairs: exact normalized name, postal
code plus a loose name token, house number plus address-token overlap,
and country plus name-prefix. A fifth, scalable fuzzy-matching pass
(sorted-neighborhood blocking, and separately a character-TF-IDF
nearest-neighbor variant) was implemented but left disabled in the final
submitted run — at this dataset's scale (5M+ records per source), the
brute-force TF-IDF variant was computationally infeasible on the
available hardware, and the scalable sorted-neighborhood variant was
deprioritized given the time remaining once the four active blocks were
confirmed working end-to-end. Embedding retrieval (e.g.
`all-MiniLM-L6-v2`) was considered as a future direction but not
implemented.

**Blocking keys used (active in the submitted run):** exact normalized
name, postal code + name token, house number + address-token overlap,
country + name-prefix. A fifth, fuzzy-matching key was implemented but
not enabled (see Section 3 above).

**Candidate pairs generated:** The final test-side candidate file
contains **32,501,248 candidate pairs**. The training-side pipeline
processed approximately **34.58 million candidate pairs**.

**Blocking recall:** **61%** on the available validation/ground-truth
evaluation.

**How true matches were protected from being lost:** Multiple blocking
rules were combined using a union rather than relying on a single key.
The resulting candidate set was deduplicated before feature extraction
and matching. The candidate file records the final set passed to the
matching model.

------------------------------------------------------------------------

## 4. Matching Model

### Features used

-   **Name features:** exact normalized-name match, suffix-stripped
    exact match, token Jaccard similarity, Levenshtein similarity,
    Jaro-Winkler similarity.
-   **Address features:** exact normalized-address match, token-set
    overlap ratio, Levenshtein similarity, postal-code match,
    house-number match.
-   **Other features:** country equality, missing-field flags (name and
    address, both sides), source indicator (Source 2 vs Source 3),
    blocking-rule-origin indicators, and three interaction features
    (name-score × address-score product; strong-name-similarity-and-
    postal-match; strong-address-similarity-and-country-match).

**Model type:** LightGBM binary classifier.

**Threshold selection method:** The decision threshold was selected
using a Source-1-held-out validation split to optimize macro $F_{0.5}$.
The selected threshold for final prediction was **0.57**.

The model was trained with early stopping, allowing training to stop
when validation performance stopped improving rather than requiring all
planned boosting rounds.

------------------------------------------------------------------------

## 5. Results & Error Analysis

**F_0.5 Score (macro):** **0.7695**

**Final decision threshold:** **0.57**

**Blocking recall:** The training pipeline reported **2,942,047 true matches that were not present in the generated candidate set** and therefore could not be recovered by the downstream classifier. This identifies candidate generation/blocking as the main measured source of recall loss in the pipeline.

**Training candidate composition:** The generated training candidate set contained **4,696,318 positive pairs** and **29,887,061 hard-negative pairs**, corresponding to a **13.58% positive rate** among candidate pairs.

### Error analysis

**1. Blocking-related false negatives**

The clearest measured failure mode was loss during candidate generation. The pipeline reported **2,942,047 true matches that were not in the candidate set**. Because the matching model only evaluates generated candidate pairs, these matches cannot be recovered regardless of the classifier or decision threshold. This indicates that improving blocking recall is a significant potential avenue for reducing false negatives.

**2. Missing address information**

Address information is incomplete in the test data. Source 2 has **2.6% missing business addresses**, while Source 3 has **2.7% missing business addresses**. Consequently, address-based similarity features cannot contribute equally to every candidate pair, increasing reliance on name, country, postal-code, and other available signals for records with missing addresses.

**3. Candidate-set class imbalance**

The training candidate set contains substantially more hard negatives than positives: **29.89 million hard negatives versus 4.70 million positives**. The resulting **13.58% positive rate** means that the classifier operates in a strongly imbalanced candidate-pair setting. The use of LightGBM followed by explicit threshold tuning addresses the need to make the final merge decision separately from candidate generation.

**4. Threshold-related trade-off**

The threshold was selected empirically rather than using the default 0.5 decision boundary. A coarse and fine threshold sweep selected **0.57**, producing a validation macro **F0.5 of 0.7695**. This indicates that the final operating point was chosen specifically for the competition metric rather than for generic classification accuracy.

**5. Test prediction outcomes**

The final test pipeline generated **32,501,248 candidate pairs** and successfully completed feature extraction across the entire candidate set. It produced predictions for all **1,732,544 Source 1 entities**, with **1,438,006 matched entities** and **294,538 predicted singletons**.
A separate quantified breakdown of individual false-positive and false-negative examples was not produced by the final run, so no specific error categories are asserted beyond the measured blocking loss and missing-data effects above.


------------------------------------------------------------------------

## 6. Conclusion

The final system separates high-recall retrieval from conservative pair
classification, which is appropriate for a precision-heavy metric and
allows one-to-many matches. The completed pipeline successfully
processed the full training candidate volume, trained a LightGBM
matcher, selected a threshold of 0.57, and generated the required test
outputs.

The achieved validation macro $F_{0.5}$ score was **0.7695**. The final
test run generated **32,501,248 candidate pairs** and produced matching
predictions for **1,732,544 Source 1 entities**.

The main empirical limitation identified in the completed evaluation is
blocking recall of **61%**. Further improvements would therefore
primarily involve expanding or refining candidate-generation rules while
preserving precision in the downstream matching model.

------------------------------------------------------------------------

## Appendix

### A. Code Artefacts

The complete, runnable code ships in the submission zip under:

`code/business_entity_resolution/`

All source code is contained under `src/`, with the project `README.md`
and `requirements.txt` included alongside it.

The main prediction entry point is:

``` powershell
cd code\business_entity_resolution\src
python main.py predict
```

The pipeline reads the configured test data, performs normalization and
blocking, extracts pairwise features, applies the trained LightGBM model
and selected threshold, and writes:

-   `output/matching_results.tsv`
-   `output/candidate_pairs.tsv`

The final submission package places these two output files at the
required top-level `output/` directory.

### B. Additional Results

The completed run produced the following operational results:

  Metric / Output                           Result
  ------------------------------- ----------------
  Validation macro $F_{0.5}$            **0.7695**
  Selected prediction threshold           **0.57**
  Blocking recall                          **61%**
  Training candidate pairs            **\~34.58M**
  Final test candidate pairs        **32,501,248**
  Source 1 entities predicted        **1,732,544**
  Matched entities                   **1,438,006**
  Predicted singleton entities         **294,538**

Feature extraction was implemented in chunks so that the full candidate
volume could be processed without exhausting available memory.

------------------------------------------------------------------------

**Team:** Ctrl Win