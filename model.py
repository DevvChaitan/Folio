"""Train and save Folio's feedback-based matching weights locally.

This script trains the feedback learner only. The Sentence Transformer remains a
pretrained text feature extractor and is not fine-tuned by this project.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from matching_engine import DEFAULT_WEIGHTS, FEATURE_NAMES, HybridMatcher, MBTI_SCORE_MATRIX
from nlp_engine import SENTENCE_TRANSFORMER_NAME, calculate_semantic_similarity, get_text_embeddings


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "Datasets"
USERS_PATH = DATA_DIR / "users.csv"
FEEDBACK_PATH = DATA_DIR / "feedback.csv"
WEIGHTS_PATH = PROJECT_DIR / "production_weights.json"

MIN_USER_FEEDBACK = 5
PERSONALIZATION_REGULARIZATION = 2.0

REQUIRED_USER_COLUMNS = {
    "user_id",
    "name",
    "location",
    "professional_summary",
    "about_me",
    "mbti",
}
REQUIRED_FEEDBACK_COLUMNS = {"user_id", "matched_user_id", "action"}


def _project_to_simplex(values: np.ndarray) -> np.ndarray:
    """Project weights to non-negative values that sum to one."""
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size != len(FEATURE_NAMES) or not np.isfinite(vector).all():
        raise ValueError("A weight vector must contain three finite values")

    ordered = np.sort(vector)[::-1]
    cumulative = np.cumsum(ordered)
    indexes = np.arange(1, vector.size + 1, dtype=np.float64)
    valid = ordered - (cumulative - 1.0) / indexes > 0.0
    if not np.any(valid):
        return np.full(vector.size, 1.0 / vector.size, dtype=np.float64)

    last_valid = int(np.flatnonzero(valid)[-1])
    threshold = (cumulative[last_valid] - 1.0) / (last_valid + 1)
    return np.maximum(vector - threshold, 0.0)


def _weights_to_dict(weights: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(FEATURE_NAMES, weights)}


def load_training_data(data_dir: str | Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the approved CSVs and reject malformed or inconsistent rows."""
    data_dir = Path(data_dir)
    users_path = data_dir / "users.csv"
    feedback_path = data_dir / "feedback.csv"
    if not users_path.is_file() or not feedback_path.is_file():
        raise FileNotFoundError(
            f"Expected users.csv and feedback.csv in {data_dir.resolve()}"
        )

    users = pd.read_csv(users_path)
    feedback = pd.read_csv(feedback_path)

    # Keep the approved feedback.csv unchanged. New app interactions are stored
    # separately and join the next local training run when that log exists.
    app_feedback_path = data_dir / "feedback_log.csv"
    if app_feedback_path.is_file():
        app_feedback = pd.read_csv(app_feedback_path)
        missing_app_columns = REQUIRED_FEEDBACK_COLUMNS - set(app_feedback.columns)
        if missing_app_columns:
            raise ValueError(
                "feedback_log.csv is missing columns: "
                f"{sorted(missing_app_columns)}"
            )
        app_feedback = app_feedback.reindex(columns=feedback.columns)
        feedback = pd.concat([feedback, app_feedback], ignore_index=True)

    missing_user_columns = REQUIRED_USER_COLUMNS - set(users.columns)
    missing_feedback_columns = REQUIRED_FEEDBACK_COLUMNS - set(feedback.columns)
    if missing_user_columns:
        raise ValueError(f"users.csv is missing columns: {sorted(missing_user_columns)}")
    if missing_feedback_columns:
        raise ValueError(f"feedback.csv is missing columns: {sorted(missing_feedback_columns)}")
    if users.empty:
        raise ValueError("users.csv has no profile rows")
    if feedback.empty:
        raise ValueError("feedback.csv has no feedback rows")
    if users["user_id"].isna().any() or feedback[["user_id", "matched_user_id"]].isna().any().any():
        raise ValueError("User IDs and feedback references cannot be empty")

    users["user_id"] = users["user_id"].astype(str).str.strip()
    feedback["user_id"] = feedback["user_id"].astype(str).str.strip()
    feedback["matched_user_id"] = feedback["matched_user_id"].astype(str).str.strip()
    if users["user_id"].eq("").any() or feedback[["user_id", "matched_user_id"]].eq("").any().any():
        raise ValueError("User IDs and feedback references cannot be blank")
    if users["user_id"].duplicated().any():
        duplicates = users.loc[users["user_id"].duplicated(), "user_id"].tolist()
        raise ValueError(f"users.csv contains duplicate user IDs: {duplicates[:5]}")

    users["mbti"] = users["mbti"].fillna("").astype(str).str.strip().str.upper()
    invalid_mbti = sorted(set(users["mbti"]) - set(MBTI_SCORE_MATRIX))
    if invalid_mbti:
        raise ValueError(f"users.csv contains invalid MBTI values: {invalid_mbti}")

    for column in ("professional_summary", "about_me"):
        users[column] = users[column].fillna("").astype(str).str.strip()
    if (users["professional_summary"].eq("") & users["about_me"].eq("")).any():
        raise ValueError("Every profile needs text in professional_summary or about_me")
    users["location"] = users["location"].fillna("").astype(str).str.strip()
    users["name"] = users["name"].fillna("").astype(str).str.strip()

    actions = pd.to_numeric(feedback["action"], errors="coerce")
    if actions.isna().any() or not actions.isin([0, 1]).all():
        raise ValueError("feedback.csv action values must be 0 (reject) or 1 (accept)")
    feedback["action"] = actions.astype(np.int64)

    known_users = set(users["user_id"])
    unknown_references = sorted(
        (set(feedback["user_id"]) | set(feedback["matched_user_id"])) - known_users
    )
    if unknown_references:
        raise ValueError(
            "feedback.csv references unknown user IDs: "
            f"{unknown_references[:10]}"
        )
    if feedback["user_id"].eq(feedback["matched_user_id"]).any():
        raise ValueError("feedback.csv contains self-match rows")

    return users.set_index("user_id", drop=False), feedback


def build_training_matrix(
    users: pd.DataFrame,
    feedback: pd.DataFrame,
    *,
    show_progress_bar: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build [text, MBTI, location] features for each viewer-match action."""
    if "user_id" not in users.columns:
        raise ValueError("users must include a user_id column")
    if users.index.name != "user_id":
        users = users.set_index("user_id", drop=False)

    profile_texts = (
        users["professional_summary"].astype(str)
        + ". "
        + users["about_me"].astype(str)
    ).tolist()
    profile_embeddings = get_text_embeddings(
        profile_texts,
        batch_size=32,
        show_progress_bar=show_progress_bar,
    )
    embeddings_by_id = {
        str(user_id): embedding
        for user_id, embedding in zip(users.index, profile_embeddings)
    }
    users_by_id: Mapping[str, Mapping[str, object]] = users.to_dict(orient="index")
    matcher = HybridMatcher(weights_path=None)

    feature_rows: list[list[float]] = []
    targets: list[float] = []
    viewer_ids: list[str] = []
    pair_feature_cache: dict[tuple[str, str], tuple[float, float, float]] = {}

    for row in feedback.itertuples(index=False):
        viewer_id = str(row.user_id)
        candidate_id = str(row.matched_user_id)
        cache_key = tuple(sorted((viewer_id, candidate_id)))
        if cache_key not in pair_feature_cache:
            viewer = users_by_id[viewer_id]
            candidate = users_by_id[candidate_id]
            pair_feature_cache[cache_key] = (
                calculate_semantic_similarity(
                    embeddings_by_id[viewer_id], embeddings_by_id[candidate_id]
                ),
                matcher.get_mbti_score(viewer["mbti"], candidate["mbti"]),
                matcher.get_location_score(viewer["location"], candidate["location"]),
            )

        feature_rows.append(list(pair_feature_cache[cache_key]))
        targets.append(float(row.action))
        viewer_ids.append(viewer_id)

    return (
        np.asarray(feature_rows, dtype=np.float64),
        np.asarray(targets, dtype=np.float64),
        np.asarray(viewer_ids, dtype=object),
    )


class CustomGradientDescent:
    """Fit interpretable mixture weights with projected gradient descent.

    Predictions are weighted averages of the three component scores, so weights
    remain non-negative, sum to one, and produce values in the 0–1 range.
    Personal models add an L2 pull toward global weights to limit overfitting.
    """

    def __init__(
        self,
        learning_rate: float = 0.05,
        epochs: int = 2000,
        regularization: float = 0.0,
        tolerance: float = 1e-9,
    ):
        if learning_rate <= 0.0 or epochs < 1 or regularization < 0.0 or tolerance < 0.0:
            raise ValueError("Invalid gradient descent configuration")
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.regularization = regularization
        self.tolerance = tolerance
        self.weights = _project_to_simplex(DEFAULT_WEIGHTS)
        self.loss_ = float("nan")
        self.n_iter_ = 0
        self.fitted_ = False

    def fit(
        self,
        X: object,
        y: object,
        *,
        initial_weights: object = None,
        prior_weights: object = None,
        verbose: bool = False,
        report_every: int = 250,
    ) -> "CustomGradientDescent":
        features = np.asarray(X, dtype=np.float64)
        targets = np.asarray(y, dtype=np.float64).reshape(-1)
        if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
            raise ValueError("X must be a matrix with text, MBTI, and location columns")
        if features.shape[0] == 0 or features.shape[0] != targets.size:
            raise ValueError("X and y must contain the same non-zero number of examples")
        if not np.isfinite(features).all() or not np.isfinite(targets).all():
            raise ValueError("Training data must contain only finite values")
        if np.any(features < 0.0) or np.any(features > 1.0):
            raise ValueError("Training features must be between 0 and 1")
        if np.any(targets < 0.0) or np.any(targets > 1.0):
            raise ValueError("Feedback targets must be between 0 and 1")

        start = DEFAULT_WEIGHTS if initial_weights is None else initial_weights
        self.weights = _project_to_simplex(np.asarray(start, dtype=np.float64))
        prior = self.weights.copy() if prior_weights is None else _project_to_simplex(
            np.asarray(prior_weights, dtype=np.float64)
        )

        for epoch in range(self.epochs):
            predictions = features @ self.weights
            errors = predictions - targets
            gradient = (2.0 / targets.size) * (features.T @ errors)
            if self.regularization:
                gradient += 2.0 * self.regularization * (self.weights - prior)

            next_weights = _project_to_simplex(
                self.weights - self.learning_rate * gradient
            )
            self.n_iter_ = epoch + 1
            if np.linalg.norm(next_weights - self.weights) <= self.tolerance:
                self.weights = next_weights
                break
            self.weights = next_weights

            if verbose and report_every > 0 and (epoch + 1) % report_every == 0:
                penalty = self.regularization * float(np.sum((self.weights - prior) ** 2))
                loss = float(np.mean((features @ self.weights - targets) ** 2) + penalty)
                print(f"Epoch {epoch + 1:4d} | objective: {loss:.6f}")

        penalty = self.regularization * float(np.sum((self.weights - prior) ** 2))
        self.loss_ = float(np.mean((features @ self.weights - targets) ** 2) + penalty)
        self.fitted_ = True
        return self

    def predict(self, X: object) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("Fit the feedback model before requesting predictions")
        features = np.asarray(X, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
            raise ValueError("X must be a matrix with text, MBTI, and location columns")
        return np.clip(features @ self.weights, 0.0, 1.0)

    def save_weights(
        self,
        path: str | Path = WEIGHTS_PATH,
        *,
        user_weights: Mapping[str, object] | None = None,
        user_sample_counts: Mapping[str, int] | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> Path:
        """Save global and personalized feedback weights to the existing JSON file."""
        if not self.fitted_:
            raise RuntimeError("Fit the feedback model before saving its weights")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        user_weight_payload = {
            str(user_id): _weights_to_dict(_project_to_simplex(np.asarray(weights)))
            for user_id, weights in (user_weights or {}).items()
        }
        artifact = {
            "schema_version": 2,
            "model_type": "simplex_constrained_feedback_regression",
            "feature_names": list(FEATURE_NAMES),
            "global_weights": _weights_to_dict(self.weights),
            "user_weights": user_weight_payload,
            "user_sample_counts": {
                str(user_id): int(count)
                for user_id, count in (user_sample_counts or {}).items()
            },
            "training_metadata": dict(metadata or {}),
        }

        temporary_path = path.with_name(f".{path.name}.tmp")
        try:
            with temporary_path.open("w", encoding="utf-8") as file:
                json.dump(artifact, file, indent=2, allow_nan=False)
                file.write("\n")
            temporary_path.replace(path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        return path


def train_feedback_model(
    data_dir: str | Path = DATA_DIR,
    weights_path: str | Path = WEIGHTS_PATH,
    *,
    show_progress_bar: bool = True,
) -> dict[str, object]:
    """Train global and per-viewer weights, then save them locally."""
    print("1. Loading and validating the approved profile and feedback CSVs.")
    users, feedback = load_training_data(data_dir)

    print(f"2. Embedding {len(users)} profiles with {SENTENCE_TRANSFORMER_NAME}.")
    print("   The text encoder stays pretrained; only feedback weights are trained.")
    X, y, viewer_ids = build_training_matrix(
        users,
        feedback,
        show_progress_bar=show_progress_bar,
    )

    print(f"3. Fitting global feedback weights from {len(y)} interactions.")
    global_learner = CustomGradientDescent(
        learning_rate=0.05,
        epochs=3000,
        regularization=0.0,
        tolerance=1e-10,
    ).fit(X, y, initial_weights=DEFAULT_WEIGHTS, verbose=True)

    print("4. Learning cautious viewer-specific adjustments.")
    user_weights: dict[str, np.ndarray] = {}
    user_sample_counts: dict[str, int] = {}
    for viewer_id in pd.unique(viewer_ids):
        mask = viewer_ids == viewer_id
        count = int(np.count_nonzero(mask))
        user_sample_counts[str(viewer_id)] = count
        if count < MIN_USER_FEEDBACK:
            continue

        personalized_learner = CustomGradientDescent(
            learning_rate=0.02,
            epochs=1000,
            regularization=PERSONALIZATION_REGULARIZATION,
            tolerance=1e-9,
        ).fit(
            X[mask],
            y[mask],
            initial_weights=global_learner.weights,
            prior_weights=global_learner.weights,
        )
        user_weights[str(viewer_id)] = personalized_learner.weights.copy()

    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_examples": int(len(y)),
        "accepted_examples": int(np.sum(y == 1.0)),
        "rejected_examples": int(np.sum(y == 0.0)),
        "viewers_with_feedback": int(len(user_sample_counts)),
        "viewers_with_personalized_weights": int(len(user_weights)),
        "minimum_feedback_for_personalization": MIN_USER_FEEDBACK,
        "personalization_regularization": PERSONALIZATION_REGULARIZATION,
        "global_training_objective": "mean_squared_error",
        "global_training_loss": float(global_learner.loss_),
        "global_training_epochs": int(global_learner.n_iter_),
        "text_encoder": SENTENCE_TRANSFORMER_NAME,
        "text_encoder_trained": False,
        "feedback_model_trained_locally": True,
    }

    saved_path = global_learner.save_weights(
        weights_path,
        user_weights=user_weights,
        user_sample_counts=user_sample_counts,
        metadata=metadata,
    )
    print(f"5. Saved feedback weights to {saved_path.resolve()}")
    print(f"   Global weights: {_weights_to_dict(global_learner.weights)}")
    print(f"   Personalized viewers: {len(user_weights)}")
    return metadata


if __name__ == "__main__":
    train_feedback_model()
