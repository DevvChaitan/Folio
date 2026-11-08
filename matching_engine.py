"""Transparent hybrid scoring for Folio profile matches."""

from __future__ import annotations

import json
import re
from itertools import product
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from nlp_engine import calculate_semantic_similarity


FEATURE_NAMES = ("text", "mbti", "location")
DEFAULT_WEIGHTS = np.asarray([0.50, 0.30, 0.20], dtype=np.float64)
DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().with_name("production_weights.json")
MBTI_TYPES = tuple("".join(chars) for chars in product("EI", "NS", "TF", "JP"))


def _build_mbti_score_matrix() -> dict[str, dict[str, float]]:
    """Build a symmetric, explainable heuristic matrix for all 16 MBTI types.

    Shared preferences receive a higher score. INTJ/ENFP is explicitly assigned
    1.0 because it is the example pairing in the project brief. This is a project
    heuristic, not a validated psychological compatibility measure.
    """
    matrix: dict[str, dict[str, float]] = {}
    for first in MBTI_TYPES:
        matrix[first] = {}
        for second in MBTI_TYPES:
            shared_preferences = sum(left == right for left, right in zip(first, second))
            score = 0.25 + 0.75 * (shared_preferences / 4.0)
            if {first, second} == {"INTJ", "ENFP"}:
                score = 1.0
            matrix[first][second] = float(score)
    return matrix


MBTI_SCORE_MATRIX = _build_mbti_score_matrix()


def _coerce_weights(weights: object) -> np.ndarray:
    """Read canonical or legacy weight data and project it to a valid simplex."""
    if isinstance(weights, Mapping):
        aliases = {"text": "w_text", "mbti": "w_mbti", "location": "w_loc"}
        values = []
        for name in FEATURE_NAMES:
            value = weights.get(name, weights.get(aliases[name]))
            if value is None:
                raise ValueError(f"Weight artifact is missing the '{name}' weight")
            values.append(float(value))
        result = np.asarray(values, dtype=np.float64)
    else:
        result = np.asarray(weights, dtype=np.float64).reshape(-1)

    if result.shape != (len(FEATURE_NAMES),) or not np.isfinite(result).all():
        raise ValueError("Weights must be three finite values for text, MBTI, and location")
    if np.any(result < 0.0):
        raise ValueError("Weights must be non-negative")
    total = float(result.sum())
    if total <= 0.0:
        raise ValueError("At least one model weight must be positive")
    return result / total


def _normalize_location(location: object) -> str:
    if location is None:
        return ""
    return re.sub(r"\s+", " ", str(location).strip()).casefold()


class HybridMatcher:
    """Score profile pairs and rank candidates using learned or default weights."""

    def __init__(self, weights_path: str | Path | None = DEFAULT_WEIGHTS_PATH):
        self.weights_path = Path(weights_path) if weights_path is not None else None
        self.weights = DEFAULT_WEIGHTS.copy()
        self.user_weights: dict[str, np.ndarray] = {}
        self.user_sample_counts: dict[str, int] = {}
        self.artifact_loaded = False

        if self.weights_path is not None and self.weights_path.exists():
            self._load_weights(self.weights_path)

    def _load_weights(self, path: Path) -> None:
        try:
            with path.open("r", encoding="utf-8") as file:
                artifact = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read Folio weight artifact at {path}: {exc}") from exc

        if not isinstance(artifact, dict):
            raise ValueError("Folio weight artifact must contain a JSON object")

        # Accept the original {w_text, w_mbti, w_loc} file for a smooth upgrade.
        global_weights = artifact.get("global_weights", artifact.get("weights", artifact))
        self.weights = _coerce_weights(global_weights)

        raw_user_weights = artifact.get("user_weights", {})
        if not isinstance(raw_user_weights, dict):
            raise ValueError("The user_weights field must be a JSON object")
        self.user_weights = {
            str(user_id): _coerce_weights(user_weights)
            for user_id, user_weights in raw_user_weights.items()
        }

        raw_counts = artifact.get("user_sample_counts", {})
        if isinstance(raw_counts, dict):
            self.user_sample_counts = {
                str(user_id): int(count) for user_id, count in raw_counts.items()
            }
        self.artifact_loaded = True

    def get_mbti_score(self, mbti1: object, mbti2: object) -> float:
        first = str(mbti1 or "").strip().upper()
        second = str(mbti2 or "").strip().upper()
        if first not in MBTI_SCORE_MATRIX or second not in MBTI_SCORE_MATRIX:
            raise ValueError("MBTI values must be valid four-letter types, such as INTJ")
        return MBTI_SCORE_MATRIX[first][second]

    def get_location_score(self, loc1: object, loc2: object) -> float:
        first = _normalize_location(loc1)
        second = _normalize_location(loc2)
        return 1.0 if first and first == second else 0.0

    @staticmethod
    def _validate_features(text_sim: object, mbti_score: object, loc_score: object) -> np.ndarray:
        features = np.asarray([text_sim, mbti_score, loc_score], dtype=np.float64)
        if not np.isfinite(features).all():
            raise ValueError("Match features must contain only finite numbers")
        if np.any(features < 0.0) or np.any(features > 1.0):
            raise ValueError("Text, MBTI, and location features must each be between 0 and 1")
        return features

    def _weights_for_user(self, viewer_id: object = None) -> tuple[np.ndarray, str]:
        if viewer_id is not None and str(viewer_id) in self.user_weights:
            return self.user_weights[str(viewer_id)], "user"
        return self.weights, "global"

    def explain_compatibility(
        self,
        text_sim: float,
        mbti_score: float,
        loc_score: float,
        *,
        viewer_id: object = None,
    ) -> dict[str, object]:
        features = self._validate_features(text_sim, mbti_score, loc_score)
        weights, weight_scope = self._weights_for_user(viewer_id)
        contributions = weights * features * 100.0
        components = {
            name: {
                "score": float(feature),
                "weight": float(weight),
                "contribution": float(contribution),
            }
            for name, feature, weight, contribution in zip(
                FEATURE_NAMES, features, weights, contributions
            )
        }
        return {
            "score": float(np.clip(contributions.sum(), 0.0, 100.0)),
            "components": components,
            "weights_scope": weight_scope,
            "viewer_id": str(viewer_id) if viewer_id is not None else None,
        }

    def calculate_compatibility(
        self,
        text_sim: float,
        mbti_score: float,
        loc_score: float,
        *,
        viewer_id: object = None,
    ) -> float:
        """Return the weighted compatibility score on a 0–100 scale."""
        return float(
            self.explain_compatibility(
                text_sim, mbti_score, loc_score, viewer_id=viewer_id
            )["score"]
        )

    def score_user_ids(
        self,
        viewer_id: object,
        candidate_id: object,
        users_by_id: Mapping[str, Mapping[str, object]],
        embeddings_by_id: Mapping[str, object],
        *,
        return_breakdown: bool = False,
    ) -> float | dict[str, object]:
        """Score two IDs using profile records and precomputed text embeddings."""
        viewer_key = str(viewer_id)
        candidate_key = str(candidate_id)
        if viewer_key == candidate_key:
            raise ValueError("A user cannot be matched with their own profile")
        if viewer_key not in users_by_id or candidate_key not in users_by_id:
            raise KeyError("Both user IDs must exist in users_by_id")
        if viewer_key not in embeddings_by_id or candidate_key not in embeddings_by_id:
            raise KeyError("Both user IDs must have precomputed text embeddings")

        viewer = users_by_id[viewer_key]
        candidate = users_by_id[candidate_key]
        text_score = calculate_semantic_similarity(
            embeddings_by_id[viewer_key], embeddings_by_id[candidate_key]
        )
        mbti_score = self.get_mbti_score(viewer.get("mbti"), candidate.get("mbti"))
        location_score = self.get_location_score(
            viewer.get("location"), candidate.get("location")
        )
        breakdown = self.explain_compatibility(
            text_score, mbti_score, location_score, viewer_id=viewer_key
        )
        return breakdown if return_breakdown else float(breakdown["score"])

    def rank_matches(
        self,
        viewer_id: object,
        candidate_ids: Iterable[object],
        users_by_id: Mapping[str, Mapping[str, object]],
        embeddings_by_id: Mapping[str, object],
        *,
        top_k: int = 5,
    ) -> list[dict[str, object]]:
        """Return the highest scoring candidates, excluding the viewer."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        viewer_key = str(viewer_id)
        ranked = []
        for candidate_id in candidate_ids:
            candidate_key = str(candidate_id)
            if candidate_key == viewer_key:
                continue
            result = self.score_user_ids(
                viewer_key,
                candidate_key,
                users_by_id,
                embeddings_by_id,
                return_breakdown=True,
            )
            candidate = users_by_id[candidate_key]
            ranked.append(
                {
                    "user_id": candidate_key,
                    "name": candidate.get("name", candidate_key),
                    **result,
                }
            )

        ranked.sort(key=lambda match: (-float(match["score"]), str(match["user_id"])))
        return ranked[:top_k]


if __name__ == "__main__":
    matcher = HybridMatcher()
    text_score = 0.4242
    mbti_score = matcher.get_mbti_score("INTJ", "ENFP")
    location_score = matcher.get_location_score("Bangalore", " bangalore ")
    breakdown = matcher.explain_compatibility(text_score, mbti_score, location_score)
    print(f"Component scores: {breakdown['components']}")
    print(f"Final compatibility score: {breakdown['score']:.2f}%")
