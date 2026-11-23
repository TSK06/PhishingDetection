"""Behavioral tests for notebook-defined dataset and evaluation helpers."""

import ast
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from sklearn.base import clone

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "PhishingDomainDetection.ipynb"
HELPER_NAMES = {
    "prepare_dataset", "split_dataset", "make_text_pipeline", "make_cv",
    "build_candidate_config", "split_summary", "metric_summary",
    "_json_safe", "_canonical_json", "_frame_fingerprint",
    "build_checkpoint_metadata", "run_resumable_cv", "evaluate_candidates",
    "evaluate_held_out_once", "_take_rows", "_emit_fit_progress",
}
HELPER_CLASS_NAMES = {"CheckpointCompatibilityError", "ResumableCheckpointStore"}


def load_notebook_helpers():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    namespace = {
        "train_test_split": train_test_split, "pd": pd, "np": np,
        "hashlib": hashlib, "json": json, "tempfile": tempfile,
        "Path": Path, "time": __import__("time"), "clone": clone,
        "ParameterGrid": ParameterGrid,
        "LABEL_MAP": {"good": 0, "bad": 1},
        "CONFLICTING_URLS": {"tommyhumphreys.com/"},
        "CountVectorizer": CountVectorizer, "Pipeline": Pipeline,
        "GridSearchCV": GridSearchCV, "StratifiedKFold": StratifiedKFold,
        "LogisticRegression": LogisticRegression, "LinearSVC": LinearSVC,
        "DecisionTreeClassifier": DecisionTreeClassifier,
        "RandomForestClassifier": RandomForestClassifier,
        "GradientBoostingClassifier": GradientBoostingClassifier,
        "XGBClassifier": XGBClassifier,
        "accuracy_score": accuracy_score,
        "precision_score": precision_score, "recall_score": recall_score,
        "f1_score": f1_score, "confusion_matrix": confusion_matrix,
        "PHISHING_LABEL": 1,
    }
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        tree = ast.parse("".join(cell.get("source", [])))
        selected = [node for node in tree.body if (
            isinstance(node, ast.FunctionDef) and node.name in HELPER_NAMES
        ) or (
            isinstance(node, ast.ClassDef) and node.name in HELPER_CLASS_NAMES
        )]
        if selected:
            module = ast.Module(body=selected, type_ignores=[])
            exec(compile(module, str(NOTEBOOK_PATH), "exec"), namespace)
    missing = (HELPER_NAMES | HELPER_CLASS_NAMES) - namespace.keys()
    if missing:
        raise AssertionError(f"Notebook evaluation helpers are missing: {missing}")
    return namespace


def sample_rows():
    rows = []
    for index in range(40):
        rows.append({"URL": f"safe-{index}.example/path", "Label": "good"})
        rows.append({"URL": f"phish-{index}.example/login", "Label": "bad"})
    rows.extend([
        {"URL": "duplicate.example/path", "Label": "good"},
        {"URL": "duplicate.example/path", "Label": "good"},
        {"URL": "tommyhumphreys.com/", "Label": "good"},
        {"URL": "tommyhumphreys.com/", "Label": "bad"},
        {"URL": "tommyhumphreys.com/", "Label": "bad"},
    ])
    return pd.DataFrame(rows)


class TrackingTokenizer:
    seen_texts = []

    def __call__(self, text):
        type(self).seen_texts.append(text)
        return text.split()


class TrackingCountVectorizer(CountVectorizer):
    fit_vocabularies = []

    def fit_transform(self, raw_documents, y=None):
        matrix = super().fit_transform(raw_documents, y)
        type(self).fit_vocabularies.append(set(self.vocabulary_))
        return matrix


class LeakageResistantEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helpers = load_notebook_helpers()

    def prepared(self):
        return self.helpers["prepare_dataset"](sample_rows())

    def test_duplicate_labels_and_class_accounting(self):
        raw = sample_rows()
        usable, conflicts, conflict_rows = self.prepared()
        raw_counts = raw["Label"].value_counts().to_dict()
        self.assertEqual(raw_counts, {"good": 43, "bad": 42})

        conflicting_counts = conflict_rows["Label"].value_counts().to_dict()
        self.assertEqual(conflicting_counts, {"bad": 2, "good": 1})
        self.assertEqual(conflicts, {"tommyhumphreys.com/"})

        group_label_counts = raw.groupby("URL")["Label"].nunique()
        inconsistent_groups = set(group_label_counts[group_label_counts > 1].index)
        self.assertEqual(inconsistent_groups, conflicts)
        self.assertTrue((group_label_counts.drop(index=list(conflicts)) == 1).all())

        unique_by_label_before_conflict = {
            label: int(raw.loc[raw["Label"] == label, "URL"].nunique())
            for label in ("good", "bad")
        }
        self.assertEqual(unique_by_label_before_conflict, {"good": 42, "bad": 41})
        final_counts = usable["Label"].value_counts().to_dict()
        self.assertEqual(final_counts, {0: 41, 1: 40})
        self.assertEqual(len(usable), 81)
        self.assertNotIn("tommyhumphreys.com/", set(usable["URL"]))

        # An additional unapproved conflict must fail instead of keeping the
        # first row or silently choosing a majority label.
        unexpected = pd.concat([raw, pd.DataFrame([
            {"URL": "safe-0.example/path", "Label": "bad"}
        ])], ignore_index=True)
        with self.assertRaises(ValueError):
            self.helpers["prepare_dataset"](unexpected)

    def test_conflicting_url_is_excluded_entirely(self):
        usable, conflicts, conflict_rows = self.prepared()
        self.assertEqual(conflicts, {"tommyhumphreys.com/"})
        self.assertEqual(conflict_rows.groupby("Label").size().to_dict(), {"bad": 2, "good": 1})
        self.assertNotIn("tommyhumphreys.com/", set(usable["URL"]))
        self.assertTrue(usable["URL"].is_unique)

    def test_exact_urls_are_isolated_across_all_partitions(self):
        usable, _, _ = self.prepared()
        train, validation, test = self.helpers["split_dataset"](usable, seed=17)
        url_sets = [set(part["URL"]) for part in (train, validation, test)]
        self.assertTrue(url_sets[0].isdisjoint(url_sets[1]))
        self.assertTrue(url_sets[0].isdisjoint(url_sets[2]))
        self.assertTrue(url_sets[1].isdisjoint(url_sets[2]))
        self.assertEqual(sum(map(len, url_sets)), len(usable))

    def test_split_sizes_and_label_distributions(self):
        usable, _, _ = self.prepared()
        train, validation, test = self.helpers["split_dataset"](usable, seed=42)
        summaries = [self.helpers["split_summary"](part) for part in (train, validation, test)]
        self.assertEqual(sum(item["rows"] for item in summaries), len(usable))
        for part, summary in zip((train, validation, test), summaries):
            counts = part["Label"].value_counts().reindex([0, 1], fill_value=0).to_dict()
            self.assertEqual(summary, {"rows": len(part), "class_counts": counts})
        self.assertAlmostEqual(len(train) / len(usable), 0.70, delta=0.02)
        self.assertAlmostEqual(len(validation) / len(usable), 0.15, delta=0.02)
        self.assertAlmostEqual(len(test) / len(usable), 0.15, delta=0.02)

    def test_same_seed_reproduces_splits(self):
        usable, _, _ = self.prepared()
        first = self.helpers["split_dataset"](usable, seed=123)
        second = self.helpers["split_dataset"](usable, seed=123)
        for left, right in zip(first, second):
            self.assertEqual(left["URL"].tolist(), right["URL"].tolist())
            self.assertEqual(left["Label"].tolist(), right["Label"].tolist())

    def test_candidate_configuration_is_reproducible_and_gradient_boosting_deferred(self):
        candidates_a, deferred_a = self.helpers["build_candidate_config"](seed=314)
        candidates_b, deferred_b = self.helpers["build_candidate_config"](seed=314)
        self.assertEqual(
            list(candidates_a),
            ["Logistic Regression", "LinearSVC", "Decision Tree", "Random Forest", "XGBoost"],
        )
        self.assertEqual(set(deferred_a), {"Gradient Boosting"})
        self.assertNotIn("Gradient Boosting", candidates_a)
        self.assertIsInstance(candidates_a["LinearSVC"][0], LinearSVC)
        self.assertNotIsInstance(candidates_a["LinearSVC"][0], SVC)
        self.assertEqual(candidates_a["LinearSVC"][1], {"C": [0.1, 1.0]})
        self.assertIsInstance(deferred_a["Gradient Boosting"][0], GradientBoostingClassifier)
        self.assertEqual(
            {name: config[1] for name, config in candidates_a.items()},
            {
                "Logistic Regression": {"C": [0.1, 1.0, 10.0]},
                "LinearSVC": {"C": [0.1, 1.0]},
                "Decision Tree": {"max_depth": [10, 25, None]},
                "Random Forest": {"n_estimators": [100], "max_depth": [None, 25]},
                "XGBoost": {"n_estimators": [100], "max_depth": [3, 6], "learning_rate": [0.1]},
            },
        )
        self.assertEqual(
            deferred_a["Gradient Boosting"][1],
            {"n_estimators": [100], "learning_rate": [0.1, 0.5], "max_depth": [2, 3]},
        )
        for name in candidates_a:
            self.assertEqual(candidates_a[name][1], candidates_b[name][1])
            self.assertEqual(
                candidates_a[name][0].get_params(deep=False),
                candidates_b[name][0].get_params(deep=False),
            )
        self.assertEqual(deferred_a["Gradient Boosting"][1], deferred_b["Gradient Boosting"][1])

    def test_cv_factory_is_fixed_three_fold_stratification(self):
        cv = self.helpers["make_cv"](seed=314)
        self.assertIsInstance(cv, StratifiedKFold)
        self.assertEqual(cv.n_splits, 3)
        self.assertTrue(cv.shuffle)
        self.assertEqual(cv.random_state, 314)

    def test_cv_fold_vectorizers_and_test_isolation(self):
        helper = self.helpers
        train_text = [f"trainword{index} shared" for index in range(12)]
        train_labels = np.array([0, 1] * 6)
        validation_text = ["validationmarker shared", "validationother shared"]
        validation_labels = np.array([1, 0])
        test_text = ["testonlymarker shared"]
        test_labels = np.array([1])
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=7)
        expected_fold_vocabularies = [
            {token for index in fold_train for token in train_text[index].split()}
            for fold_train, _ in cv.split(train_text, train_labels)
        ]
        TrackingCountVectorizer.fit_vocabularies = []
        TrackingTokenizer.seen_texts = []
        tokenizer = TrackingTokenizer()
        candidates = {
            "constant phishing": (
                DummyClassifier(strategy="constant", constant=1),
                {"constant": [1]},
            )
        }
        with tempfile.TemporaryDirectory(dir=NOTEBOOK_PATH.parent) as temp_dir:
            metadata = helper["build_checkpoint_metadata"](
                "synthetic-dataset", train_text, train_labels,
                validation_text, validation_labels, 7, candidates,
                {"tokenizer": "tracking", "vectorizer": "count"},
            )
            store = helper["ResumableCheckpointStore"](temp_dir, metadata)
            events = []
            cv_results = helper["run_resumable_cv"](
                candidates, train_text, train_labels, cv, tokenizer, store,
                seed=7, vectorizer_class=TrackingCountVectorizer,
                planned_total_fits=4, progress_callback=events.append,
            )
            rows, selected, selected_name, selected_parameters, _ = helper["evaluate_candidates"](
                candidates, cv_results, train_text, train_labels,
                validation_text, validation_labels, tokenizer, store, 4,
                vectorizer_class=TrackingCountVectorizer,
            )
        self.assertEqual(selected_name, "constant phishing")
        self.assertEqual(selected_parameters, {"constant": 1})
        self.assertEqual(len(rows), 1)
        self.assertEqual(TrackingCountVectorizer.fit_vocabularies[:3], expected_fold_vocabularies)
        self.assertEqual(len(TrackingCountVectorizer.fit_vocabularies), 4)  # 3 folds + refit
        self.assertTrue(any("validationmarker" in text for text in TrackingTokenizer.seen_texts))
        self.assertFalse(any("testonlymarker" in text for text in TrackingTokenizer.seen_texts))
        self.assertEqual(len(events), 3)
        self.assertEqual([event["overall_completed"] for event in events], [1, 2, 3])
        self.assertTrue(all(event["planned_total_fits"] == 4 for event in events))
        self.assertEqual([event["remaining_fits"] for event in events], [3, 2, 1])
        self.assertTrue(all(event["candidate"] == "constant phishing" for event in events))
        self.assertTrue(all(event["fit_duration_seconds"] >= 0 for event in events))
        self.assertTrue(all(event["cumulative_elapsed_seconds"] >= 0 for event in events))
        self.assertTrue(all(event["failed_fit_count"] == 0 for event in events))

        fitted_vocab = dict(selected.named_steps["vectorizer"].vocabulary_)
        predict_calls = []

        class CountingEstimator:
            def predict(self, values):
                predict_calls.append(list(values))
                return selected.predict(values)

        with tempfile.TemporaryDirectory(dir=NOTEBOOK_PATH.parent) as temp_dir:
            metadata = helper["build_checkpoint_metadata"](
                "synthetic-dataset", train_text, train_labels,
                validation_text, validation_labels, 7, candidates,
                {"tokenizer": "tracking", "vectorizer": "count"},
            )
            store = helper["ResumableCheckpointStore"](temp_dir, metadata)
            store.finalize_selection("constant phishing", selected_parameters, rows[0]["validation"])
            predictions, test_metrics = helper["evaluate_held_out_once"](
                CountingEstimator(), test_text, test_labels, store,
                selected_name, selected_parameters,
            )
            cached_predictions, cached_metrics = helper["evaluate_held_out_once"](
                None, test_text, test_labels, store, selected_name, selected_parameters,
            )
        self.assertEqual(predictions.tolist(), [1])
        self.assertIsNone(cached_predictions)
        self.assertEqual(test_metrics["phishing_recall"], 1.0)
        self.assertEqual(cached_metrics, test_metrics)
        self.assertEqual(len(predict_calls), 1)
        self.assertEqual(len(TrackingCountVectorizer.fit_vocabularies), 4)
        self.assertEqual(selected.named_steps["vectorizer"].vocabulary_, fitted_vocab)
        self.assertTrue(any("testonlymarker" in text for text in TrackingTokenizer.seen_texts))

    def _small_search_fixture(self, temp_dir, seed=7):
        train = [f"trainword{index} shared" for index in range(18)]
        labels = np.array([0, 1] * 9)
        validation = ["validationmarker shared", "validationother shared"]
        validation_labels = np.array([1, 0])
        candidates = {"constant phishing": (
            DummyClassifier(strategy="constant", constant=1), {"constant": [1]}
        )}
        metadata = self.helpers["build_checkpoint_metadata"](
            "synthetic-dataset", train, labels, validation, validation_labels,
            seed, candidates, {"tokenizer": "tracking", "vectorizer": "count"},
        )
        store = self.helpers["ResumableCheckpointStore"](temp_dir, metadata)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        return train, labels, validation, validation_labels, candidates, store, cv

    def test_completed_folds_are_checkpointed_and_skipped_on_resume(self):
        helper = self.helpers
        TrackingCountVectorizer.fit_vocabularies = []
        with tempfile.TemporaryDirectory(dir=NOTEBOOK_PATH.parent) as temp_dir:
            train, labels, _, _, candidates, store, cv = self._small_search_fixture(temp_dir)
            events_first, events_second = [], []
            first = helper["run_resumable_cv"](
                candidates, train, labels, cv, TrackingTokenizer(), store, seed=7,
                vectorizer_class=TrackingCountVectorizer, progress_callback=events_first.append,
            )
            fit_count = len(TrackingCountVectorizer.fit_vocabularies)
            second = helper["run_resumable_cv"](
                candidates, train, labels, cv, TrackingTokenizer(), store, seed=7,
                vectorizer_class=TrackingCountVectorizer, progress_callback=events_second.append,
            )
            self.assertEqual(fit_count, 3)
            self.assertEqual(len(TrackingCountVectorizer.fit_vocabularies), fit_count)
            self.assertEqual(first, second)
            self.assertEqual(len(events_first), 3)
            self.assertTrue(all(event["status"] == "checkpoint_reused" for event in events_second))
            self.assertEqual(store.count_completed_fold_records(), 3)

    def test_incomplete_and_failed_folds_are_rerun(self):
        helper = self.helpers
        TrackingCountVectorizer.fit_vocabularies = []
        with tempfile.TemporaryDirectory(dir=NOTEBOOK_PATH.parent) as temp_dir:
            train, labels, _, _, candidates, store, cv = self._small_search_fixture(temp_dir)
            parameters, seed = {"constant": 1}, 7
            metrics = {"accuracy": 0.5, "phishing_precision": 0.5,
                       "phishing_recall": 0.5, "phishing_f1": 0.5,
                       "confusion_matrix": [[1, 1], [1, 1]]}
            store.save_fold("constant phishing", parameters, 1, seed, "complete", metrics, 0.01)
            store.save_fold("constant phishing", parameters, 2, seed, "running")
            store.save_fold("constant phishing", parameters, 3, seed, "failed",
                            error={"type": "SyntheticFailure", "message": "retry"})
            helper["run_resumable_cv"](
                candidates, train, labels, cv, TrackingTokenizer(), store, seed=seed,
                vectorizer_class=TrackingCountVectorizer,
            )
            self.assertEqual(len(TrackingCountVectorizer.fit_vocabularies), 2)
            for fold in (1, 2, 3):
                self.assertIsNotNone(store.load_completed_fold(
                    "constant phishing", parameters, fold, seed
                ))

    def test_fit_exception_is_persisted_and_reraised(self):
        helper = self.helpers

        class FailingClassifier(BaseEstimator, ClassifierMixin):
            def fit(self, features, labels):
                raise ValueError("synthetic estimator failure")

            def predict(self, features):
                return np.zeros(len(features), dtype=int)

        with tempfile.TemporaryDirectory(dir=NOTEBOOK_PATH.parent) as temp_dir:
            train = [f"trainword{index} shared" for index in range(18)]
            labels = np.array([0, 1] * 9)
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=7)
            candidates = {"broken": (FailingClassifier(), {})}
            metadata = helper["build_checkpoint_metadata"](
                "synthetic-dataset", train, labels, ["validation"], [1], 7,
                candidates, {"tokenizer": "tracking", "vectorizer": "count"},
            )
            store = helper["ResumableCheckpointStore"](temp_dir, metadata)
            with self.assertRaisesRegex(ValueError, "synthetic estimator failure"):
                helper["run_resumable_cv"](
                    candidates, train, labels, cv, TrackingTokenizer(), store, seed=7,
                )
            failure_path = store.fold_path("broken", {}, 1, 7)
            failure = json.loads(failure_path.read_text(encoding="utf-8"))
            self.assertEqual(failure["status"], "failed")
            self.assertEqual(failure["error"]["type"], "ValueError")
            self.assertIn("synthetic estimator failure", failure["error"]["message"])

    def test_checkpoint_rejects_changed_split_dataset_and_candidates(self):
        helper = self.helpers
        with tempfile.TemporaryDirectory(dir=NOTEBOOK_PATH.parent) as temp_dir:
            train, labels, validation, val_labels, candidates, store, _ = self._small_search_fixture(temp_dir)
            changed_split = helper["build_checkpoint_metadata"](
                "synthetic-dataset", train, labels, validation, val_labels,
                8, candidates, {"tokenizer": "tracking", "vectorizer": "count"},
            )
            with self.assertRaisesRegex(helper["CheckpointCompatibilityError"], "mismatch"):
                helper["ResumableCheckpointStore"](temp_dir, changed_split)
            for key, replacement in (
                ("dataset_identity", "different-dataset"),
                ("candidate_configuration", [{"name": "different"}]),
                ("preprocessing_configuration", {"tokenizer": "changed"}),
            ):
                changed = dict(store.compatibility)
                changed[key] = replacement
                with self.assertRaisesRegex(helper["CheckpointCompatibilityError"], "mismatch"):
                    helper["ResumableCheckpointStore"](temp_dir, changed)

    def test_validation_checkpoints_skip_completed_validation_work(self):
        helper = self.helpers
        TrackingCountVectorizer.fit_vocabularies = []
        TrackingTokenizer.seen_texts = []
        with tempfile.TemporaryDirectory(dir=NOTEBOOK_PATH.parent) as temp_dir:
            train, labels, validation, val_labels, candidates, store, cv = self._small_search_fixture(temp_dir)
            cv_results = helper["run_resumable_cv"](
                candidates, train, labels, cv, TrackingTokenizer(), store, seed=7,
            )
            args = (candidates, cv_results, train, labels, validation, val_labels,
                    TrackingTokenizer(), store, 4)
            first = helper["evaluate_candidates"](*args)
            seen_validation = sum("validationmarker" in text for text in TrackingTokenizer.seen_texts)
            self.assertEqual(store.count_completed_validation_records(), 1)
            second = helper["evaluate_candidates"](*args)
            self.assertEqual(first[0], second[0])
            self.assertEqual(store.count_completed_validation_records(), 1)
            self.assertEqual(seen_validation,
                             sum("validationmarker" in text for text in TrackingTokenizer.seen_texts))

    def test_search_and_validation_helpers_have_no_test_inputs(self):
        import inspect
        for helper_name in ("run_resumable_cv", "evaluate_candidates"):
            parameters = inspect.signature(self.helpers[helper_name]).parameters
            self.assertFalse(any("test" in name.lower() for name in parameters))

    def test_test_evaluation_is_blocked_until_selection_finalized(self):
        helper = self.helpers
        with tempfile.TemporaryDirectory(dir=NOTEBOOK_PATH.parent) as temp_dir:
            _, _, _, _, _, store, _ = self._small_search_fixture(temp_dir)
            calls = []

            class CountingEstimator:
                def predict(self, values):
                    calls.append(1)
                    return np.ones(len(values), dtype=int)

            with self.assertRaisesRegex(RuntimeError, "selection is finalized"):
                helper["evaluate_held_out_once"](
                    CountingEstimator(), ["testmarker"], [1], store,
                    "constant phishing", {"constant": 1},
                )
            self.assertEqual(calls, [])
            store.finalize_selection("constant phishing", {"constant": 1},
                                     {"phishing_recall": 1.0})
            predictions, _ = helper["evaluate_held_out_once"](
                CountingEstimator(), ["testmarker"], [1], store,
                "constant phishing", {"constant": 1},
            )
            self.assertEqual(predictions.tolist(), [1])
            self.assertEqual(calls, [1])

    def test_vectorizer_learns_only_from_training_text(self):
        pipeline = self.helpers["make_text_pipeline"](
            str.split, DummyClassifier(strategy="most_frequent")
        )
        train_text = ["train uniquealpha", "train uniquebeta"]
        pipeline.fit(train_text, [0, 1])
        vectorizer = pipeline.named_steps["vectorizer"]
        vocabulary = dict(vectorizer.vocabulary_)
        self.assertEqual(set(vocabulary), {"train", "uniquealpha", "uniquebeta"})
        pipeline.predict(["validation onlygamma", "test onlydelta"])
        self.assertEqual(vectorizer.vocabulary_, vocabulary)
        self.assertNotIn("onlygamma", vectorizer.vocabulary_)
        self.assertNotIn("onlydelta", vectorizer.vocabulary_)


if __name__ == "__main__":
    unittest.main()
