##  California Housing Price Prediction

##  Objective
Predict median house values (`MedHouseVal`) in California districts using regression models.

##  Project Structure
- `eda.ipynb` — Exploratory Data Analysis & outlier detection.
- `model.ipynb` — Preprocessing, feature scaling, training & evaluation (`MAE`, `RMSE`, `R²`).
- `app.py` — Interactive Streamlit prediction app.
- `requirements.txt` — Project dependencies.

##  les moyens d'evaluation
- MAE
- RMSE
- R² Score
## les models 
- Linear Regression
- Ridge Regression
- Random Forest

## ⚙️ From Experimentation to Production (`train.py`)

While `model.ipynb` is used for experimentation and benchmarking multiple algorithms (`Linear`, `Ridge`, `Lasso`, and `Random Forest`), **`train.py`** serves as the standalone production training script:

- **Why `train.py`?** It converts the winning workflow into modular, reproducible Python code that runs end-to-end with a single command, avoids Jupyter hidden-state issues, and is automatically linted via GitHub Actions (`flake8`).

- **Why only Random Forest?** Since `Random Forest` clearly outperformed all linear baselines in `model.ipynb` (**RMSE: `0.5122`**, **MAE: `0.3325`**, **$R^2$: `0.7998`**), `train.py` trains and serializes only the winning `RobustScaler + RandomForestRegressor` pipeline (`models/best_model.joblib`) to keep retraining fast and deployment-ready.


### Run Training Pipeline
```bash
python 01-California-Housing-Prediction/train.py