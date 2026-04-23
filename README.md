# IMDb Movie Success Analysis

An end-to-end IMDb analytics and machine learning project structured like a small production application. It combines a reusable data pipeline, a FastAPI backend, a polished Streamlit dashboard, richer model comparison workflows, cached model artifacts, live row ingestion, and targeted tests in one portfolio-ready codebase.

## Project Objectives

- Build a robust pipeline for loading, cleaning, and preparing large IMDb datasets.
- Explore which metadata patterns are associated with highly rated and widely voted titles.
- Compare multiple machine learning models for a binary movie success prediction task.
- Expose analytics and prediction functionality through a backend API.
- Present the results in a polished, interactive dashboard with reusable components.

## Architecture

```text
imdb-movie-success-analysis-upgraded/
+-- app/
¦   +-- backend/
¦   ¦   +-- routes/
¦   ¦   +-- schemas/
¦   ¦   +-- main.py
¦   +-- config/
¦   +-- dashboard/
¦   ¦   +-- components/
¦   +-- services/
¦   +-- utils/
+-- assets/
+-- data/
¦   +-- processed/
¦   +-- raw/
+-- models/
¦   +-- artifacts/
+-- notebooks/
+-- tests/
+-- api.py
+-- dashboard.py
+-- README.md
+-- requirements.txt
```

## Dataset

This project uses the [IMDb Dataset on Kaggle](https://www.kaggle.com/datasets/ashirwadsangwan/imdb-dataset), primarily:

- `title.basics.tsv`
- `title.ratings.tsv`

The application filters to relevant title types such as movies, TV series, mini-series, TV movies, specials, and direct-to-video content. The prepared dataset includes:

- title identifiers and titles
- title type
- release year and decade
- runtime
- genres and main genre
- IMDb average rating
- vote count and log-transformed vote count
- derived rating and vote buckets
- `success_score = averageRating * log10(numVotes + 1)` for balancing quality and audience reach

## Core Features

### Data Pipeline

- chunked reading for large IMDb basics files
- safe missing value handling and numeric coercion
- merge validation between basics and ratings
- engineered features such as `mainGenre`, `decade`, `logVotes`, `success_score`, and bucketed rating/vote segments
- prepared dataset persistence to `data/processed/`
- live row validation for uploaded/manual additions inside the dashboard

### Dashboard

- sidebar filters for year, rating, votes, genre, type, search, highlight title, and model mode
- saved filter presets for common analytical views
- KPI cards and executive summary callouts
- key findings section with auto-generated filtered insights
- interactive Plotly charts with consistent dark-mode styling
- a consolidated `Analysis & Visualizations` tab that groups overview, content type, popularity, pairwise, genre, trend, explorer, and data-quality analysis into one organized workspace
- a richer `Model Lab` with full model selection, per-model diagnostics, confusion matrix, ROC support, threshold analysis, calibration, importance views, and dynamic model-weight controls
- a dedicated `Prediction Sandbox` tab with sample input loading, reset actions, multi-model comparison, weighted ensemble probability, and clear input/output explanations
- a new `About` tab with project context, objectives, dataset background, technologies, model explanations, and usage guidance
- live data ingestion from CSV/TSV uploads and manual title entry
- weighted ensemble ranking driven by normalized model weights
- CSV downloads for filtered data and model results
- dedicated data-quality analysis panel with missingness, outliers, suspicious patterns, and recommendations

### Backend API

Implemented endpoints include:

- `GET /health`
- `GET /health/details`
- `GET /movies`
- `GET /movies/{id}`
- `GET /analytics/summary`
- `GET /analytics/genres`
- `GET /analytics/trends`
- `POST /predict`
- `GET /models/compare`

The API supports:

- filtering and search
- pagination and sorting for `/movies`
- CSV export from `/movies` and analytics endpoints
- details on processed dataset files and model artifacts

### Machine Learning

The project frames success as:

`success = 1` if `averageRating >= rating_threshold` and `numVotes >= votes_threshold`

Models compared:

- Logistic Regression
- Logistic Regression (unbalanced baseline)
- Decision Tree
- Random Forest
- Random Forest (unbalanced baseline)
- Gradient Boosting
- K-Nearest Neighbors
- tuned top models in full mode

Evaluation includes:

- accuracy
- precision
- recall
- f1-score
- ROC-AUC
- confusion matrix
- ROC curve
- cross-validation
- threshold tuning
- calibration curve
- simple error analysis
- feature importance and permutation importance
- class imbalance comparison

## Methodology

### Preprocessing

The data pipeline filters to the most relevant title types, merges basics and ratings on `tconst`, converts year/runtime/rating/vote columns into clean numerics, drops invalid target rows, and fills selected missing values in a controlled way. Additional features such as `mainGenre`, `decade`, `logVotes`, `success_score`, and bucketed categorical summaries are created to improve downstream analysis.

### Feature Selection

The predictive pipeline uses only:

- `startYear`
- `runtimeMinutes`
- `isAdult`
- `titleType`
- `genres`

These features are transformed with scaling, one-hot encoding, and custom multi-label genre binarization.

### Leakage Prevention

`averageRating` and `numVotes` are used to define the success target. Because of that, they are intentionally excluded from model predictors to avoid target leakage. This is one of the most important methodological safeguards in the project.

### Model Comparison Strategy

The dashboard and API compare multiple baseline models, evaluate them with holdout metrics and cross-validation, and in full mode run lightweight hyperparameter tuning on the top-performing candidates. The final comparison includes both performance and interpretability outputs.

The updated dashboard also lets the analyst assign normalized weights to models and use those weights in a weighted ensemble score and weighted sandbox prediction.

## Results Summary

A typical comparison table in the project reports:

| Model | What It Usually Shows |
|---|---|
| Logistic Regression | Strong interpretable baseline and good calibration |
| Decision Tree | Easier to inspect but less stable |
| Random Forest | Often best balance of non-linearity and robustness |
| Gradient Boosting | Competitive performance with good ranking ability |
| KNN | Useful contrast model but can struggle at scale |

The actual best model depends on the current success definition and active filters. The dashboard now includes a dedicated results summary table with an interpretation column, rather than only raw metrics.

### Why The Best Model Won

The project now generates an explicit explanation of why the best model wins, using:

- holdout F1 and ROC-AUC
- cross-validation stability
- feature importance
- permutation importance
- observed dataset patterns in title type, year, and genres

This gives the model comparison more academic value than simply declaring a winner from one score.

## Assumptions and Limitations

- The raw IMDb files are large, so first-run preparation and first-run full modeling can take noticeable time.
- Only `title.basics` and `title.ratings` are used; adding `title.crew` or `title.principals` would enable richer modeling.
- The current predictor focuses on structured metadata available in the selected IMDb tables, so it cannot capture marketing, distribution, cast quality, or review-text effects.
- The success definition is analyst-controlled and therefore subjective; changing the thresholds changes the class boundary and can change which model performs best.
- The dataset is naturally biased toward titles with IMDb coverage and audience voting behavior, which may underrepresent smaller or less visible releases.
- Live uploaded/manual rows are excellent for demo and exploratory use, but they depend on the user providing sensible values.
- Prepared dataset persistence prefers Parquet and falls back to pickle if Parquet support is unavailable.

## Setup

### 1. Create or activate your virtual environment

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Place IMDb data files

Put these files in either:

- `data/raw/`
- or the project root

Files required:

- `title.basics.tsv` or `title.basics.tsv.gz`
- `title.ratings.tsv` or `title.ratings.tsv.gz`

### 4. Prepare the dataset

```bash
python notebooks/improved_data_loading.py --force
```

This creates a cached prepared dataset in `data/processed/`.

## Running the Project

### Run the dashboard

```bash
streamlit run dashboard.py
```

Then open the local Streamlit URL shown in the terminal, typically `http://localhost:8501`.

### Run the backend

```bash
uvicorn app.backend.main:app --reload
```

Open the API documentation at:

- `http://127.0.0.1:8000/docs`

## Example API Usage

### Health check

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/health/details
```

### Filter movies with pagination

```bash
curl "http://127.0.0.1:8000/movies?min_rating=7.5&min_votes=50000&page=1&limit=20&sort_by=numVotes"
```

### Export genres as CSV

```bash
curl "http://127.0.0.1:8000/analytics/genres?top_n=20&export_format=csv"
```

### Compare models in fast mode

```bash
curl "http://127.0.0.1:8000/models/compare?success_rating=7.2&success_votes=30000&mode=fast"
```

### Predict success

```bash
curl -X POST "http://127.0.0.1:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"titleType\":\"movie\",\"startYear\":2024,\"runtimeMinutes\":115,\"isAdult\":0,\"genres\":[\"Drama\",\"Thriller\"]}"
```

## Tests

Run the verification suite with:

```bash
pytest
```

## Main Files Added or Refactored

- `app/services/data_loader.py`: shared loading, cleaning, filtering, live-row preparation, and caching
- `app/services/analytics.py`: analytics summaries, key findings, and recommendation-style insights
- `app/services/data_quality.py`: missingness, outlier, suspicious pattern, and quality assessment helpers
- `app/services/modeling.py`: model training, comparison, caching, per-model diagnostics, weighted multi-model prediction, and normalized weight helpers
- `app/backend/main.py`: FastAPI app
- `app/backend/routes/`: API endpoints with pagination and export support
- `app/dashboard/main.py`: modular Streamlit entry point
- `app/dashboard/components/analysis.py`: unified analysis-and-visualizations tab
- `app/dashboard/components/about.py`: project background and usage guide tab
- `app/dashboard/components/modeling.py`: model lab and separate prediction sandbox UI
- `app/dashboard/components/`: reusable dashboard sections, filters, layout, and modeling views
- `assets/dashboard.css`: dashboard styling system
- `tests/`: analytics, data quality, API, and modeling verification

## Future Improvements

- integrate crew/cast features for stronger predictive power
- add experiment tracking and model versioning metadata UI
- expose data-quality summaries through dedicated API endpoints
- add deployment configuration for API and dashboard hosting
- add CI for tests and linting



