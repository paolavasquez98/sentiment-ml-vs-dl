# IMDB Sentiment Analysis

We compare TF-IDF + Logistic Regression (with Multinomial Naive Bayes as a second baseline) against fine-tuned DistilBERT on IMDB movie reviews.

## Running the notebook

Use Python 3.12 or newer. Run `uv sync`, then open `sent_analysis.ipynb` with the project environment and run the cells in order from this directory. The first run downloads the labelled Hugging Face training split. The unsupervised split is not used.

The notebook keeps the original step-by-step structure: preprocessing and EDA, frozen splits, baseline models, preprocessing ablations, and dev evaluation. DistilBERT training and the final test evaluation remain the next stage.

## Data and evaluation rules

- Seed: **42**. Labels: 0 = negative, 1 = positive.
- Strip HTML, lowercase and collapse whitespace for exact normalized-text deduplication before splitting. This does not detect semantic near-duplicates.
- Hold out exactly **5,000** stratified dev reviews from the deduplicated official training split. The remaining reviews form the training split.
- `data/frozen_splits.csv` stores original HF row IDs, labels, source text SHA-256 hashes, split membership and seed. Keep this file in version control and share it between model implementations. Reruns reload and verify it; do not delete or regenerate it once experiments begin.
- Fit vocabulary, IDF weights and all models on training reviews only. Choose Logistic Regression C from 0.1, 1 and 10 on dev accuracy; keep C fixed in the preprocessing ablations.
- **Load and evaluate the official 25,000-review test split only once, after all configurations are frozen at the planned day-7 gate.** Do not inspect it during development or tune on its results. The current notebook does not load it.
- Experiment B uses 1,000, 5,000 and the full available training split, not 25,000: deduplication and the dev holdout reduce the available training count. Save identical subset IDs for both models before starting this experiment.

## Results and hand-off

Each notebook execution creates a unique folder in `results/`. It saves every model/configuration's dev predictions, metrics, confidence buckets, confusion matrices, a Logistic Regression error sample and two independent annotation templates. Runs never overwrite previous predictions. Keep notebook outputs cleared when committing.

Prediction CSV columns include `model`, `config`, `seed`, `example_id`, `label`, `pred`, `prob` and `split_sha256`. **`prob` is P(positive)**. DistilBERT must use the same example IDs, split hash and probability convention. Configuration JSON records preprocessing and model settings; `metadata.json` records package versions, the dataset fingerprint and split sizes. Fit time includes vectorizer fitting and classifier training, excluding dev inference and downloads.

The baseline keeps stopwords. The rating-removal and English-stopword ablations change one factor at a time. All results at this stage are dev diagnostics, not final generalization estimates or causal proof of label leakage.

Next: fine-tune DistilBERT on a GPU; run truncation and matched training-size experiments; freeze configurations; save paired test predictions; compute 95% bootstrap accuracy intervals and McNemar's test; independently annotate errors from both model families and report reviewer disagreement. Human annotations must not be filled automatically.

## Validated baseline results

The notebook was executed through the baseline stage on 26 October 2026. Normalized-text deduplication removed 98 reviews, leaving 19,902 training and 5,000 dev examples; 8.70% of deduplicated reviews contain an explicit numeric rating. Saved results are in `results/20261026T124628Z_8dea6956/`.

| Configuration | Dev accuracy | Dev macro-F1 |
| --- | ---: | ---: |
| Logistic Regression, C = 10 | 90.12% | 0.9012 |
| Multinomial Naive Bayes | 88.28% | 0.8828 |
| Logistic Regression, numeric ratings removed | 90.12% | 0.9012 |
| Logistic Regression, English stopwords removed | 89.36% | 0.8936 |

Keeping stopwords is supported by this dev comparison. Removing numeric ratings leaves aggregate accuracy unchanged on this run; this does not imply that individual predictions are identical or that rating shortcuts never matter. These results were selected and inspected on dev. DistilBERT, the two main experiments and final test statistics are still outstanding.
