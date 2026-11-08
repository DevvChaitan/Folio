# Folio

**Live app:** [folioengine.streamlit.app](https://folioengine.streamlit.app/)

Folio is a hybrid profile-matching recommendation system. It compares professional and personal profile text, MBTI preferences, and location to rank potential connections. Each result includes a score breakdown so users can see why it was recommended.

Accept and Skip actions feed a local feedback loop that adjusts matching weights for each viewer. Folio is a project prototype; recommendation improvements should be claimed only after evaluation on later, held-out feedback.

## How it works

1. **Text understanding:** Folio cleans profile text and encodes it with the pretrained `sentence-transformers/all-MiniLM-L6-v2` model. Cosine similarity measures semantic similarity between profiles.
2. **Hybrid scoring:** Semantic similarity, an explicit MBTI compatibility heuristic, and normalized location matching are combined into a 0-100 score.
3. **Recommendations:** The app ranks the top five candidates, excludes the active profile, and explains each score by component.
4. **Feedback learning:** Accept/Skip feedback is stored separately in `Datasets/feedback_log.csv`; the supplied `users.csv` and `feedback.csv` remain unchanged. The learner updates global and viewer-specific weights and saves them in `production_weights.json`.

## Open-weight text model

Folio uses the pretrained [all-MiniLM-L6-v2 model](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), which produces 384-dimensional sentence embeddings and is published under the Apache-2.0 license. Folio uses it as a fixed text encoder; the feedback learner trains the matching weights locally.

## Tech stack

- Python, Pandas, and NumPy
- Streamlit for the interactive app and Plotly for charts
- Sentence Transformers, Transformers, and PyTorch for profile embeddings
- scikit-learn is included in the environment for machine-learning utilities; the feedback-weight optimizer is implemented with NumPy gradient descent

## Run locally

From the project folder, create and activate an environment, install dependencies, then launch the app:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app reads the approved CSVs from `Datasets/`. On first use, the pretrained text model may need to download if it is not already cached.

## Train feedback weights locally

```powershell
python model.py
```

This trains the feedback-weight model from the profile and feedback CSVs, plus `Datasets/feedback_log.csv` when that app-generated log exists. The fitted global and per-viewer weights and training metadata are saved to `production_weights.json`. The text encoder is not fine-tuned.
