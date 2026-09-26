"""
CLI entry point.

    python main.py train      -> full pipeline, trains + saves the model, reports validation F0.5
    python main.py evaluate   -> trains + runs the threshold sweep (no model save)
    python main.py predict    -> loads the saved model, runs test-time inference, writes outputs
"""

import argparse

import config
import io_utils
import normalize
import blocking
import labels
import features as feat
import model
import evaluate
import predict


def _load_and_prep_train(use_cache: bool = True):
    import pipeline

    s1n, s2n, s3n, candidates, gt = pipeline.load_and_prepare(use_cache=use_cache)
    labeled, missed = labels.build_pairwise_labels(candidates, gt)
    labels.summarize_labels(labeled, missed)

    featured = feat.extract_features(labeled, s1n, s2n, s3n)
    return gt, featured


def cmd_train(use_cache: bool = True):
    gt, featured = _load_and_prep_train(use_cache=use_cache)
    train_ids, valid_ids = model.split_entity_ids(gt)
    train_df = model.filter_by_entity_ids(featured, train_ids)
    valid_df = model.filter_by_entity_ids(featured, valid_ids)

    trained_model = model.train_lightgbm(train_df, valid_df)
    model.save_model(trained_model)

    valid_df = valid_df.copy()
    valid_df["match_probability"] = model.predict_proba(trained_model, valid_df)
    result = evaluate.evaluate_at_threshold(valid_df, gt, valid_ids, config.DEFAULT_THRESHOLD)
    print(f"\nValidation macro F0.5 @ threshold={config.DEFAULT_THRESHOLD}: {result['macro_f0.5']:.4f}")


def cmd_evaluate(use_cache: bool = True):
    gt, featured = _load_and_prep_train(use_cache=use_cache)
    train_ids, valid_ids = model.split_entity_ids(gt)
    train_df = model.filter_by_entity_ids(featured, train_ids)
    valid_df = model.filter_by_entity_ids(featured, valid_ids)

    trained_model = model.train_lightgbm(train_df, valid_df)
    valid_df = valid_df.copy()
    valid_df["match_probability"] = model.predict_proba(trained_model, valid_df)

    print("\n--- Threshold sweep (macro F0.5) ---")
    for t in config.THRESHOLD_SWEEP:
        result = evaluate.evaluate_at_threshold(valid_df, gt, valid_ids, t)
        print(f"threshold={t:.2f}  macro_F0.5={result['macro_f0.5']:.4f}")


def cmd_predict():
    predict.run_test_pipeline()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Business entity resolution pipeline")
    parser.add_argument("command", choices=["train", "evaluate", "predict"])
    parser.add_argument("--no-cache", action="store_true", help="Recompute normalize+blocking instead of using the cache")
    args = parser.parse_args()

    if args.command == "train":
        cmd_train(use_cache=not args.no_cache)
    elif args.command == "evaluate":
        cmd_evaluate(use_cache=not args.no_cache)
    elif args.command == "predict":
        cmd_predict()