"""Folio: a local, interactive profile-matching demo built with Streamlit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from matching_engine import HybridMatcher, MBTI_TYPES
from model import (
    DATA_DIR,
    PERSONALIZATION_REGULARIZATION,
    WEIGHTS_PATH,
    CustomGradientDescent,
    load_training_data,
)
from nlp_engine import get_text_embeddings


st.set_page_config(
    page_title="Folio | Meaningful matches",
    page_icon="F",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEEDBACK_LOG_PATH = DATA_DIR / "feedback_log.csv"
FEEDBACK_LOG_COLUMNS = (
    "user_id",
    "matched_user_id",
    "action",
    "timestamp",
    "text_sim",
    "mbti_score",
    "location_score",
)


def load_feedback_log() -> list[dict[str, Any]]:
    """Read app-generated feedback without changing the approved feedback.csv."""
    if not FEEDBACK_LOG_PATH.is_file():
        return []
    try:
        saved = pd.read_csv(FEEDBACK_LOG_PATH)
        required = set(FEEDBACK_LOG_COLUMNS)
        if not required.issubset(saved.columns):
            return []
        events = []
        for row in saved.itertuples(index=False):
            features = [float(row.text_sim), float(row.mbti_score), float(row.location_score)]
            if not np.isfinite(features).all() or int(row.action) not in (0, 1):
                continue
            events.append(
                {
                    "viewer_id": str(row.user_id),
                    "matched_user_id": str(row.matched_user_id),
                    "action": int(row.action),
                    "timestamp": str(row.timestamp),
                    "features": features,
                    "persisted": True,
                }
            )
        return events
    except (OSError, ValueError, AttributeError, pd.errors.ParserError):
        return []


def save_feedback_log(events: list[dict[str, Any]]) -> None:
    """Atomically save durable profile feedback to its own local CSV."""
    rows = []
    for event in events:
        if not event.get("persisted"):
            continue
        features = event["features"]
        rows.append(
            {
                "user_id": event["viewer_id"],
                "matched_user_id": event["matched_user_id"],
                "action": int(event["action"]),
                "timestamp": event["timestamp"],
                "text_sim": float(features[0]),
                "mbti_score": float(features[1]),
                "location_score": float(features[2]),
            }
        )

    FEEDBACK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = FEEDBACK_LOG_PATH.with_name(".feedback_log.csv.tmp")
    try:
        pd.DataFrame(rows, columns=FEEDBACK_LOG_COLUMNS).to_csv(temporary_path, index=False)
        temporary_path.replace(FEEDBACK_LOG_PATH)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


@st.cache_data(show_spinner="Preparing Folio's profile index…")
def load_app_data() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Load the approved CSVs and create reusable profile embeddings."""
    users, feedback = load_training_data(DATA_DIR)
    users = users.reset_index(drop=True)
    profile_texts = (
        users["professional_summary"].astype(str)
        + ". "
        + users["about_me"].astype(str)
    ).tolist()
    embeddings = get_text_embeddings(profile_texts, show_progress_bar=False)
    return users, feedback, embeddings


@st.cache_data(ttl=30, show_spinner=False)
def load_model_metadata() -> dict[str, Any]:
    if not Path(WEIGHTS_PATH).is_file():
        return {}
    try:
        with Path(WEIGHTS_PATH).open("r", encoding="utf-8") as file:
            artifact = json.load(file)
        return artifact.get("training_metadata", {})
    except (OSError, json.JSONDecodeError, AttributeError):
        return {}


def get_matcher() -> HybridMatcher:
    """Keep a matcher per browser session so temporary feedback stays private."""
    if "folio_matcher" not in st.session_state:
        st.session_state.folio_matcher = HybridMatcher()
        st.session_state.folio_base_global_weights = st.session_state.folio_matcher.weights.copy()
        st.session_state.folio_base_user_weights = {
            user_id: weights.copy()
            for user_id, weights in st.session_state.folio_matcher.user_weights.items()
        }
    else:
        st.session_state.setdefault(
            "folio_base_global_weights", st.session_state.folio_matcher.weights.copy()
        )
        st.session_state.setdefault(
            "folio_base_user_weights",
            {
                user_id: weights.copy()
                for user_id, weights in st.session_state.folio_matcher.user_weights.items()
            },
        )
    return st.session_state.folio_matcher


def initialize_session() -> None:
    st.session_state.setdefault("light_theme", False)
    if "feedback_log" not in st.session_state:
        st.session_state.feedback_log = load_feedback_log()
    st.session_state.setdefault("active_profile_id", None)
    st.session_state.setdefault("active_profile", None)
    st.session_state.setdefault("active_embedding", None)
    st.session_state.setdefault("feedback_notice", None)


def apply_theme(light_theme: bool) -> dict[str, str]:
    if light_theme:
        palette = {
            "background": "#f3f6fb",
            "surface": "#ffffff",
            "surface_alt": "#edf2f8",
            "text": "#162238",
            "muted": "#65748b",
            "border": "rgba(33, 55, 84, .12)",
            "accent": "#6254e8",
            "accent_soft": "#eeedff",
            "cyan": "#0f9eaa",
            "shadow": "0 14px 35px rgba(35, 54, 84, .08)",
            "plotly": "plotly_white",
        }
    else:
        palette = {
            "background": "#0b1220",
            "surface": "#121d2e",
            "surface_alt": "#17253a",
            "text": "#eef3fb",
            "muted": "#9aabc2",
            "border": "rgba(171, 190, 216, .14)",
            "accent": "#9a8cff",
            "accent_soft": "#28264a",
            "cyan": "#5bd3cf",
            "shadow": "0 18px 45px rgba(0, 0, 0, .22)",
            "plotly": "plotly_dark",
        }

    st.markdown(
        f"""
        <style>
          :root {{ color-scheme: {'light' if light_theme else 'dark'}; }}
          [data-testid="stAppViewContainer"] {{
            background:
              radial-gradient(circle at 85% 2%, {palette['accent_soft']} 0, transparent 25%),
              {palette['background']};
            color: {palette['text']};
          }}
          [data-testid="stHeader"] {{ background: transparent; }}
          [data-testid="stSidebar"] {{
            background: {palette['surface']};
            border-right: 1px solid {palette['border']};
          }}
          [data-testid="stSidebar"] * {{ color: {palette['text']}; }}
          h1, h2, h3, h4, p, label {{ color: {palette['text']}; }}
          .stCaption, [data-testid="stCaptionContainer"] {{ color: {palette['muted']}; }}
          [data-testid="stMetric"] {{
            background: {palette['surface']};
            border: 1px solid {palette['border']};
            border-radius: 18px;
            padding: 16px 18px;
            box-shadow: {palette['shadow']};
          }}
          [data-testid="stMetricLabel"] {{ color: {palette['muted']}; }}
          [data-testid="stMetricValue"] {{ color: {palette['text']}; }}
          div[data-testid="stVerticalBlockBorderWrapper"] > div {{
            border-color: {palette['border']} !important;
            border-radius: 18px !important;
          }}
          div[data-testid="stButton"] > button,
          div[data-testid="stFormSubmitButton"] > button {{
            border-radius: 12px;
            border: 1px solid {palette['border']};
            min-height: 42px;
            font-weight: 650;
            transition: transform .15s ease, box-shadow .15s ease;
          }}
          div[data-testid="stButton"] > button:hover,
          div[data-testid="stFormSubmitButton"] > button:hover {{
            transform: translateY(-1px);
            box-shadow: {palette['shadow']};
            border-color: {palette['accent']};
          }}
          div[data-testid="stTextInput"] input,
          div[data-testid="stTextArea"] textarea {{
            border-radius: 12px;
            border-color: {palette['border']};
          }}
          .folio-eyebrow {{
            color: {palette['accent']};
            font-size: .78rem;
            font-weight: 800;
            letter-spacing: .14em;
            text-transform: uppercase;
            margin-bottom: .35rem;
          }}
          .folio-hero {{
            background: linear-gradient(125deg, {palette['surface']}, {palette['surface_alt']});
            border: 1px solid {palette['border']};
            border-radius: 24px;
            padding: 28px 30px;
            box-shadow: {palette['shadow']};
            margin: .3rem 0 1.2rem 0;
          }}
          .folio-hero h1 {{ margin: 0; font-size: clamp(2rem, 4vw, 3.2rem); }}
          .folio-hero p {{ color: {palette['muted']}; margin: .55rem 0 0 0; max-width: 760px; }}
          .folio-chip {{
            display: inline-block;
            background: {palette['accent_soft']};
            color: {palette['accent']};
            border: 1px solid {palette['border']};
            border-radius: 999px;
            padding: 5px 11px;
            font-size: .78rem;
            font-weight: 700;
            margin-right: 6px;
          }}
          .folio-sidebar-brand {{
            text-align: center;
            padding: 14px 8px 18px;
          }}
          .folio-brand-mark {{
            display: grid;
            place-items: center;
            width: 46px;
            height: 46px;
            margin: 0 auto 10px;
            border-radius: 16px;
            color: #ffffff !important;
            background: linear-gradient(135deg, {palette['accent']}, {palette['cyan']});
            box-shadow: {palette['shadow']};
            font-size: 1.25rem;
            font-weight: 850;
          }}
          .folio-brand-title {{
            color: {palette['text']} !important;
            font-size: 1.6rem;
            font-weight: 850;
            letter-spacing: -.045em;
            line-height: 1.1;
          }}
          .folio-brand-subtitle {{
            color: {palette['muted']} !important;
            font-size: .86rem;
            line-height: 1.5;
            margin-top: 8px;
          }}
          .folio-sidebar-card {{
            background: linear-gradient(135deg, {palette['surface_alt']}, {palette['surface']});
            border: 1px solid {palette['border']};
            border-radius: 18px;
            padding: 15px 16px;
            margin-top: 12px;
          }}
          .folio-sidebar-card-kicker {{
            color: {palette['accent']} !important;
            font-size: .68rem;
            font-weight: 850;
            letter-spacing: .13em;
            margin-bottom: 6px;
          }}
          .folio-sidebar-card-text {{
            color: {palette['text']} !important;
            font-size: .9rem;
            font-weight: 650;
            line-height: 1.45;
          }}
          .folio-muted {{ color: {palette['muted']}; }}
          hr {{ border-color: {palette['border']}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    return palette


def _chart_template(palette: dict[str, str]) -> str:
    return palette["plotly"]


def _make_profile_maps(
    users: pd.DataFrame, embeddings: np.ndarray
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    users_by_id = users.set_index("user_id", drop=False).to_dict(orient="index")
    embeddings_by_id = {
        str(user_id): vector for user_id, vector in zip(users["user_id"], embeddings)
    }
    return users_by_id, embeddings_by_id


def _record_feedback(
    viewer_id: str,
    candidate_id: str,
    action: int,
    feature_values: list[float],
    *,
    persist: bool,
) -> str:
    """Upsert a feedback label and persist it for saved profiles."""
    events = list(st.session_state.feedback_log)
    events = [
        event
        for event in events
        if not (
            event["viewer_id"] == viewer_id
            and event["matched_user_id"] == candidate_id
        )
    ]
    events.append(
        {
            "viewer_id": viewer_id,
            "matched_user_id": candidate_id,
            "action": int(action),
            "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
            "features": [float(value) for value in feature_values],
            "persisted": bool(persist),
        }
    )
    st.session_state.feedback_log = events
    if persist:
        save_feedback_log(events)
        load_app_data.clear()
        load_model_metadata.clear()
        return "Saved locally. Your approved feedback.csv remains unchanged."
    return "Recorded for this temporary profile in this browser session only."


def apply_feedback_adaptation(
    matcher: HybridMatcher,
    events: list[dict[str, Any]],
) -> None:
    """Rebuild session weights from the saved base model plus feedback events."""
    global_prior = st.session_state.folio_base_global_weights.copy()
    base_user_weights = st.session_state.folio_base_user_weights
    matcher.user_weights = {
        user_id: weights.copy() for user_id, weights in base_user_weights.items()
    }

    grouped: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        grouped.setdefault(str(event["viewer_id"]), []).append(event)

    for viewer_id, viewer_events in grouped.items():
        initial_weights = base_user_weights.get(viewer_id, global_prior)
        learner = CustomGradientDescent(
            learning_rate=0.02,
            epochs=750,
            regularization=PERSONALIZATION_REGULARIZATION,
            tolerance=1e-9,
        ).fit(
            np.asarray([event["features"] for event in viewer_events], dtype=np.float64),
            np.asarray([event["action"] for event in viewer_events], dtype=np.float64),
            initial_weights=initial_weights,
            prior_weights=global_prior,
        )
        matcher.user_weights[viewer_id] = learner.weights.copy()


def _profile_label(profile: dict[str, Any]) -> str:
    name = profile.get("name") or profile.get("user_id", "Profile")
    profession = profile.get("profession") or "Professional"
    location = profile.get("location") or "Location not set"
    return f"{name}  |  {profession}  |  {location}"


def _show_match_studio(
    users: pd.DataFrame,
    feedback: pd.DataFrame,
    embeddings: np.ndarray,
    matcher: HybridMatcher,
    palette: dict[str, str],
) -> None:
    st.markdown(
        f"""
        <div class="folio-hero">
          <div class="folio-eyebrow">Match studio</div>
          <h1>Find people who fit your next chapter.</h1>
          <p>Explore thoughtful professional and personal matches, see how each score is built, and record what feels right.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    users_by_id, embeddings_by_id = _make_profile_maps(users, embeddings)
    profile_mode = st.radio(
        "Start with",
        ["Choose a profile", "Create a temporary profile"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if profile_mode == "Choose a profile":
        user_ids = list(users["user_id"].astype(str))
        selected_id = st.selectbox(
            "Choose a profile",
            options=user_ids,
            format_func=lambda user_id: _profile_label(users_by_id[user_id]),
            label_visibility="collapsed",
        )
        chosen = users_by_id[selected_id]
        preview_left, preview_right = st.columns([1.6, 1], gap="large")
        with preview_left:
            st.markdown(f"### {chosen.get('name', selected_id)}")
            st.caption(
                f"{chosen.get('profession', 'Professional')}  |  "
                f"{chosen.get('experience_years', '?')} years' experience"
            )
            summary = chosen.get("professional_summary", "")
            st.write(summary[:420] + ("…" if len(summary) > 420 else ""))
        with preview_right:
            st.markdown("**Profile signals**")
            st.markdown(
                f"<span class='folio-chip'>{chosen.get('mbti', 'MBTI not set')}</span>"
                f"<span class='folio-chip'>{chosen.get('location', 'Location not set')}</span>",
                unsafe_allow_html=True,
            )
            st.caption("Your saved profile stays in the approved local dataset.")
        if st.button("Find my matches", type="primary", use_container_width=True):
            st.session_state.active_profile_id = selected_id
            st.session_state.active_profile = dict(chosen)
            st.session_state.active_embedding = embeddings_by_id[selected_id]
            st.session_state.feedback_notice = None
    else:
        st.markdown("#### Build a session-only profile")
        with st.form("temporary_profile_form", clear_on_submit=False):
            name_col, profession_col = st.columns(2)
            with name_col:
                name = st.text_input("Name", placeholder="Your name")
            with profession_col:
                profession = st.text_input("Profession", placeholder="e.g. Product designer")
            location_col, mbti_col = st.columns(2)
            with location_col:
                location = st.text_input("Location", placeholder="City")
            with mbti_col:
                mbti = st.selectbox("MBTI", options=MBTI_TYPES, index=0)
            professional_summary = st.text_area(
                "Professional summary",
                placeholder="Your goals, experience, and the work you want to do…",
                height=110,
            )
            about_me = st.text_area(
                "About me",
                placeholder="How you like to collaborate and what matters outside work…",
                height=110,
            )
            submitted = st.form_submit_button(
                "Create profile and find matches",
                type="primary",
                use_container_width=True,
            )
        if submitted:
            profile_text = f"{professional_summary}. {about_me}".strip(" .")
            if not profile_text:
                st.error("Add a professional summary or a little about yourself first.")
            else:
                temporary_id = f"SESSION_PROFILE_{uuid4().hex[:8]}"
                temporary_profile = {
                    "user_id": temporary_id,
                    "name": name.strip() or "Your temporary profile",
                    "profession": profession.strip() or "Professional",
                    "location": location.strip(),
                    "mbti": mbti,
                    "professional_summary": professional_summary.strip(),
                    "about_me": about_me.strip(),
                }
                temporary_embedding = get_text_embeddings([profile_text])[0]
                st.session_state.active_profile_id = temporary_id
                st.session_state.active_profile = temporary_profile
                st.session_state.active_embedding = temporary_embedding
                st.session_state.feedback_notice = None
                st.success("Temporary profile created. It will stay in this browser session.")

    active_id = st.session_state.active_profile_id
    active_profile = st.session_state.active_profile
    if not active_id or not active_profile:
        st.info("Choose a saved profile or create a temporary one to see matches.")
        return

    if str(active_id).startswith("SESSION_PROFILE_"):
        users_by_id[active_id] = active_profile
        embeddings_by_id[active_id] = st.session_state.active_embedding

    if active_id not in users_by_id or active_id not in embeddings_by_id:
        st.warning("That profile is not available in the current dataset. Please select it again.")
        return

    candidate_ids = [user_id for user_id in users_by_id if user_id != active_id]
    try:
        matches = matcher.rank_matches(
            active_id,
            candidate_ids,
            users_by_id,
            embeddings_by_id,
            top_k=5,
        )
    except (KeyError, ValueError) as error:
        st.error(f"Folio couldn't score this profile: {error}")
        return

    if not matches:
        st.warning("No candidate profiles are available yet.")
        return

    if st.session_state.feedback_notice:
        st.success(st.session_state.feedback_notice)
        st.session_state.feedback_notice = None

    st.markdown("## Your top matches")
    st.caption(
        f"Personalization: {'viewer-specific' if matches[0]['weights_scope'] == 'user' else 'global'} weights  |  "
        "Select a score explanation to see how each signal contributed."
    )

    chart_frame = pd.DataFrame(
        {
            "Candidate": [f"{match['name']}  |  {match['user_id']}" for match in matches],
            "Compatibility": [float(match["score"]) for match in matches],
        }
    )
    chart = px.bar(
        chart_frame.sort_values("Compatibility"),
        x="Compatibility",
        y="Candidate",
        orientation="h",
        text=chart_frame.sort_values("Compatibility")["Compatibility"].map(lambda value: f"{value:.0f}%"),
        color="Compatibility",
        color_continuous_scale=[palette["cyan"], palette["accent"]],
        range_x=[0, 100],
    )
    chart.update_layout(
        template=_chart_template(palette),
        height=300,
        showlegend=False,
        coloraxis_showscale=False,
        margin=dict(l=8, r=16, t=12, b=8),
        xaxis_title="Compatibility score",
        yaxis_title=None,
    )
    chart.update_traces(textposition="outside", cliponaxis=False)
    st.plotly_chart(chart, use_container_width=True, config={"displayModeBar": False})

    for rank, match in enumerate(matches, start=1):
        candidate_id = str(match["user_id"])
        candidate = users_by_id[candidate_id]
        with st.container(border=True):
            top_left, top_middle, top_right = st.columns([0.18, 1.55, 0.5], gap="medium")
            with top_left:
                st.markdown(f"### {rank:02d}")
            with top_middle:
                st.markdown(f"### {candidate.get('name', candidate_id)}")
                st.caption(
                    f"{candidate.get('profession', 'Professional')}  |  "
                    f"{candidate.get('location', 'Location not set')}  |  "
                    f"{candidate.get('experience_years', '?')} years' experience"
                )
            with top_right:
                st.metric("Compatibility", f"{float(match['score']):.1f}%")

            profile_summary = str(candidate.get("professional_summary", ""))
            st.write(profile_summary[:320] + ("…" if len(profile_summary) > 320 else ""))

            previous_feedback = next(
                (
                    event
                    for event in st.session_state.feedback_log
                    if event["viewer_id"] == str(active_id)
                    and event["matched_user_id"] == candidate_id
                ),
                None,
            )
            if previous_feedback:
                if int(previous_feedback["action"]) == 1:
                    st.success("Accepted. Your choice is recorded.")
                else:
                    st.warning("Skipped. Folio will remember this choice.")

            with st.expander("Why this match?", expanded=False):
                component_data = match["components"]
                breakdown = pd.DataFrame(
                    [
                        {
                            "Signal": label,
                            "Signal score": float(component_data[key]["score"]) * 100,
                            "Weight": float(component_data[key]["weight"]) * 100,
                            "Contribution": float(component_data[key]["contribution"]),
                        }
                        for key, label in (
                            ("text", "Semantic text"),
                            ("mbti", "MBTI pairing"),
                            ("location", "Location"),
                        )
                    ]
                )
                contribution_chart = px.bar(
                    breakdown.sort_values("Contribution"),
                    x="Contribution",
                    y="Signal",
                    orientation="h",
                    text=breakdown.sort_values("Contribution")["Contribution"].map(
                        lambda value: f"{value:.1f}"
                    ),
                    color="Signal",
                    color_discrete_sequence=[palette["accent"], palette["cyan"], "#f0aa62"],
                )
                contribution_chart.update_layout(
                    template=_chart_template(palette),
                    height=205,
                    showlegend=False,
                    margin=dict(l=8, r=28, t=8, b=8),
                    xaxis_title="Points contributed to the 100-point score",
                    yaxis_title=None,
                )
                contribution_chart.update_traces(textposition="outside", cliponaxis=False)
                st.plotly_chart(
                    contribution_chart,
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
                st.dataframe(
                    breakdown.set_index("Signal").style.format(
                        {
                            "Signal score": "{:.1f}%",
                            "Weight": "{:.1f}%",
                            "Contribution": "{:.2f} pts",
                        }
                    ),
                    use_container_width=True,
                )

            accept_col, reject_col, feedback_caption = st.columns([0.22, 0.22, 1.2])
            features = [
                float(component_data[key]["score"])
                for key in ("text", "mbti", "location")
            ]
            with accept_col:
                if st.button("Accept", key=f"accept_{active_id}_{candidate_id}", use_container_width=True):
                    st.session_state.feedback_notice = _record_feedback(
                        str(active_id),
                        candidate_id,
                        1,
                        features,
                        persist=not str(active_id).startswith("SESSION_PROFILE_"),
                    )
                    st.rerun()
            with reject_col:
                if st.button("Skip", key=f"reject_{active_id}_{candidate_id}", use_container_width=True):
                    st.session_state.feedback_notice = _record_feedback(
                        str(active_id),
                        candidate_id,
                        0,
                        features,
                        persist=not str(active_id).startswith("SESSION_PROFILE_"),
                    )
                    st.rerun()
            with feedback_caption:
                if str(active_id).startswith("SESSION_PROFILE_"):
                    st.caption("Temporary-profile feedback stays in this browser session.")
                else:
                    st.caption("Feedback is saved locally and gently adapts this viewer's ranking.")

    session_events = st.session_state.feedback_log
    if session_events:
        st.markdown("### Recorded app feedback")
        session_frame = pd.DataFrame(session_events)
        display_frame = session_frame.drop(columns=["features"], errors="ignore")
        st.dataframe(display_frame, use_container_width=True, hide_index=True)
        st.download_button(
            "Download local feedback log CSV",
            data=display_frame.to_csv(index=False).encode("utf-8"),
            file_name="folio_feedback_log.csv",
            mime="text/csv",
        )


def _show_model_insights(
    users: pd.DataFrame,
    feedback: pd.DataFrame,
    matcher: HybridMatcher,
    palette: dict[str, str],
) -> None:
    st.markdown(
        """
        <div class="folio-hero">
          <div class="folio-eyebrow">Model insights</div>
          <h1>See what Folio is learning.</h1>
          <p>Inspect the local feedback model, its learned weights, and the approved interactions behind this demo.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    metadata = load_model_metadata()
    accept_count = int((feedback["action"].astype(int) == 1).sum())
    reject_count = int((feedback["action"].astype(int) == 0).sum())
    accept_rate = 100 * accept_count / max(1, len(feedback))
    model_rows = len(matcher.user_weights)

    metric_cols = st.columns(4, gap="medium")
    metric_cols[0].metric("Profiles", f"{len(users):,}")
    metric_cols[1].metric("Feedback signals", f"{len(feedback):,}")
    metric_cols[2].metric("Accept rate", f"{accept_rate:.1f}%")
    metric_cols[3].metric("Personalized viewers", f"{model_rows:,}")

    weights_frame = pd.DataFrame(
        {
            "Signal": ["Semantic text", "MBTI pairing", "Location"],
            "Weight": [float(value) * 100 for value in matcher.weights],
        }
    )
    weights_chart = px.bar(
        weights_frame,
        x="Signal",
        y="Weight",
        text=weights_frame["Weight"].map(lambda value: f"{value:.1f}%"),
        color="Signal",
        color_discrete_sequence=[palette["accent"], palette["cyan"], "#f0aa62"],
    )
    weights_chart.update_layout(
        template=_chart_template(palette),
        height=320,
        showlegend=False,
        yaxis=dict(range=[0, max(70, float(weights_frame["Weight"].max()) + 10)]),
        margin=dict(l=12, r=12, t=20, b=12),
        yaxis_title="Share of the final score",
        xaxis_title=None,
    )
    weights_chart.update_traces(textposition="outside")

    actions_frame = pd.DataFrame(
        {"Feedback": ["Accepted", "Rejected"], "Count": [accept_count, reject_count]}
    )
    actions_chart = px.pie(
        actions_frame,
        names="Feedback",
        values="Count",
        hole=0.68,
        color="Feedback",
        color_discrete_map={"Accepted": palette["cyan"], "Rejected": "#ee8e8e"},
    )
    actions_chart.update_layout(
        template=_chart_template(palette),
        height=320,
        margin=dict(l=12, r=12, t=20, b=12),
        legend=dict(orientation="h", y=-0.12),
    )

    left_chart, right_chart = st.columns([1.3, 0.9], gap="large")
    with left_chart:
        st.markdown("### Global score weights")
        st.plotly_chart(weights_chart, use_container_width=True, config={"displayModeBar": False})
    with right_chart:
        st.markdown("### Feedback balance")
        st.plotly_chart(actions_chart, use_container_width=True, config={"displayModeBar": False})

    st.markdown("### Feedback over time")
    if "timestamp" in feedback.columns:
        dated = feedback.copy()
        dated["timestamp"] = pd.to_datetime(dated["timestamp"], errors="coerce", utc=True)
        dated = dated.dropna(subset=["timestamp"])
        if not dated.empty:
            dated["week"] = dated["timestamp"].dt.tz_convert(None).dt.to_period("W").astype(str)
            dated["action_label"] = dated["action"].astype(int).map({1: "Accepted", 0: "Rejected"})
            by_week = (
                dated.groupby(["week", "action_label"], as_index=False)
                .size()
                .rename(columns={"size": "Interactions"})
            )
            timeline = px.line(
                by_week,
                x="week",
                y="Interactions",
                color="action_label",
                markers=True,
                color_discrete_map={"Accepted": palette["cyan"], "Rejected": "#ee8e8e"},
            )
            timeline.update_layout(
                template=_chart_template(palette),
                height=300,
                margin=dict(l=12, r=12, t=15, b=12),
                xaxis_title=None,
                legend_title=None,
            )
            st.plotly_chart(timeline, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("No valid timestamps are available to chart.")
    else:
        st.info("The feedback file has no timestamp column yet.")

    if matcher.user_weights:
        st.markdown("### How personalized weights vary")
        weight_rows = [
            {"Viewer": user_id, **{key: float(value) for key, value in zip(("text", "mbti", "location"), weights)}}
            for user_id, weights in matcher.user_weights.items()
        ]
        personalized = pd.DataFrame(weight_rows).melt(
            id_vars="Viewer", var_name="Signal", value_name="Weight"
        )
        personalized["Weight"] *= 100
        distribution = px.box(
            personalized,
            x="Signal",
            y="Weight",
            color="Signal",
            points="outliers",
            color_discrete_sequence=[palette["accent"], palette["cyan"], "#f0aa62"],
        )
        distribution.update_layout(
            template=_chart_template(palette),
            height=310,
            showlegend=False,
            margin=dict(l=12, r=12, t=15, b=12),
            xaxis_title=None,
            yaxis_title="Weight (%)",
        )
        st.plotly_chart(distribution, use_container_width=True, config={"displayModeBar": False})

    baseline_mse = (accept_count / max(1, len(feedback))) * (reject_count / max(1, len(feedback)))
    trained_loss = metadata.get("global_training_loss")
    trained_at = metadata.get("trained_at_utc", "Not recorded")
    st.markdown("### Training record")
    detail_left, detail_right = st.columns(2)
    with detail_left:
        st.markdown(f"**Last trained (UTC):** {trained_at}")
        st.markdown(f"**Feedback model trained locally:** {'Yes' if metadata.get('feedback_model_trained_locally') else 'Unknown'}")
        st.markdown(f"**Text encoder fine-tuned:** {'Yes' if metadata.get('text_encoder_trained') else 'No'}")
    with detail_right:
        if trained_loss is not None:
            st.metric("Training MSE", f"{float(trained_loss):.6f}", help="In-sample objective; it is not a holdout score.")
        st.metric("Constant accept-rate MSE", f"{baseline_mse:.6f}")
    st.info(
        "The training loss is only an in-sample fit measure. Use later interactions as a chronological holdout before claiming that feedback improved recommendations."
    )

    if st.session_state.feedback_log:
        session_frame = pd.DataFrame(st.session_state.feedback_log).drop(
            columns=["features"], errors="ignore"
        )
        st.markdown("### Feedback captured by this app")
        st.dataframe(session_frame, use_container_width=True, hide_index=True)
        st.download_button(
            "Download local feedback log CSV",
            data=session_frame.to_csv(index=False).encode("utf-8"),
            file_name="folio_feedback_log.csv",
            mime="text/csv",
        )


def main() -> None:
    initialize_session()
    with st.sidebar:
        st.markdown(
            """
            <div class="folio-sidebar-brand">
              <div class="folio-brand-mark">F</div>
              <div class="folio-brand-title">Folio</div>
              <div class="folio-brand-subtitle">Thoughtful connections,<br>explained clearly.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("EXPLORE")
        page = st.radio(
            "Workspace",
            ["Match studio", "Model insights"],
            label_visibility="collapsed",
        )
        st.divider()
        st.toggle("Bright mode", key="light_theme", help="Switch between dark and bright themes.")
        if st.button("Refresh match model", use_container_width=True):
            st.session_state.pop("folio_matcher", None)
            st.session_state.pop("folio_base_global_weights", None)
            st.session_state.pop("folio_base_user_weights", None)
            load_model_metadata.clear()
            st.rerun()
        st.divider()
        st.markdown(
            """
            <div class="folio-sidebar-card">
              <div class="folio-sidebar-card-kicker">MAKE IT MEANINGFUL</div>
              <div class="folio-sidebar-card-text">Good connections start with one thoughtful match.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    palette = apply_theme(bool(st.session_state.light_theme))
    matcher = get_matcher()
    try:
        users, feedback, embeddings = load_app_data()
    except Exception as error:
        st.error(f"Folio could not load its local data or text encoder: {error}")
        st.info("Check the local Python environment, approved CSV files, and cached Sentence Transformer.")
        st.stop()

    apply_feedback_adaptation(matcher, st.session_state.feedback_log)

    with st.container():
        st.markdown(
            f"<div class='folio-eyebrow'>Folio | {len(users)} profiles | {len(feedback)} feedback signals</div>",
            unsafe_allow_html=True,
        )

    if page == "Match studio":
        _show_match_studio(users, feedback, embeddings, matcher, palette)
    else:
        _show_model_insights(users, feedback, matcher, palette)

    st.markdown("---")
    st.caption(
        "Match scores are guides, not guarantees. MBTI pairing is a transparent heuristic, not a psychological assessment."
    )


if __name__ == "__main__":
    main()
