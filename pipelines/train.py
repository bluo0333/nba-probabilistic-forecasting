from pathlib import Path
import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss


TRAIN_TEST_SPLIT_DATE = "2018-01-01"
TRAIN_START_DATE = "2010-01-01"
RECENCY_HALFLIFE_YEARS = 5

FEATURES = [
    "home_elo_pre",
    "away_elo_pre",
    "home_avg_pts_for_last5",
    "home_avg_pts_against_last5",
    "away_avg_pts_for_last5",
    "away_avg_pts_against_last5",
    "home_win_pct_last5",
    "away_win_pct_last5",
    "home_rest_days",
    "away_rest_days",
    "home_b2b",
    "away_b2b",
    "home_netrtg_last10",
    "away_netrtg_last10",
    "home_efg_last10",
    "home_tov_pct_last10",
    "home_orb_pct_last10",
    "home_ftr_last10",
    "away_efg_last10",
    "away_tov_pct_last10",
    "away_orb_pct_last10",
    "away_ftr_last10",
    "net_diff_last5",
    "win_pct_diff_last5",
    "elo_diff",
    "rest_diff",
    "b2b_diff",
    "netrtg_diff_last10",
    "efg_diff_last10",
    "tov_pct_diff_last10",
    "orb_pct_diff_last10",
    "ftr_diff_last10",
]


def build_model(c_value: float) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("logit", LogisticRegression(max_iter=3000, C=c_value, solver="lbfgs")),
        ]
    )


def make_recency_weights(dates: pd.Series, halflife_years: float) -> np.ndarray:
    max_date = dates.max()
    ages_in_years = (max_date - dates).dt.days / 365.25
    return np.power(0.5, ages_in_years / halflife_years).to_numpy()


def tune_c_with_time_series_cv(X: pd.DataFrame, y: pd.Series, dates: pd.Series) -> float:
    candidate_c = [0.1, 0.2, 0.5, 1.0, 2.0]
    tscv = TimeSeriesSplit(n_splits=5)
    y_array = y.to_numpy()
    best_c = candidate_c[0]
    best_score = float("inf")

    for c_value in candidate_c:
        fold_losses = []
        for train_idx, val_idx in tscv.split(X):
            X_tr = X.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_tr = y_array[train_idx]
            y_val = y_array[val_idx]
            w_tr = make_recency_weights(dates.iloc[train_idx], RECENCY_HALFLIFE_YEARS)

            model = build_model(c_value)
            model.fit(X_tr, y_tr, logit__sample_weight=w_tr)
            probs = model.predict_proba(X_val)[:, 1]
            fold_losses.append(log_loss(y_val, probs))

        mean_loss = float(np.mean(fold_losses))
        print(f"CV C={c_value}: mean log loss={mean_loss:.6f}")
        if mean_loss < best_score:
            best_score = mean_loss
            best_c = c_value

    print(f"Selected C via time-series CV: {best_c}")
    return best_c


def main():
    repo_root = Path(__file__).resolve().parents[1]
    db_path = repo_root / "data" / "nba.duckdb"
    model_dir = repo_root / "models"
    model_dir.mkdir(exist_ok=True)

    conn = duckdb.connect(str(db_path))

    print("Loading features...")
    df = conn.execute("SELECT * FROM final_features").df()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["game_date", "game_id"]).reset_index(drop=True)

    train = df[(df["game_date"] < TRAIN_TEST_SPLIT_DATE) & (df["game_date"] >= TRAIN_START_DATE)]
    test = df[df["game_date"] >= TRAIN_TEST_SPLIT_DATE]

    X_train = train[FEATURES]
    y_train = train["home_win"]

    X_test = test[FEATURES]
    y_test = test["home_win"]

    print(f"Training rows: {len(train)} | Test rows: {len(test)}")
    print("Tuning hyperparameters with rolling time-series CV...")
    best_c = tune_c_with_time_series_cv(X_train, y_train, train["game_date"])

    print("Training model...")
    model = build_model(best_c)
    sample_weight = make_recency_weights(train["game_date"], RECENCY_HALFLIFE_YEARS)
    model.fit(X_train, y_train, logit__sample_weight=sample_weight)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    print("Log Loss:", log_loss(y_test, probs))
    print("Brier Score:", brier_score_loss(y_test, probs))
    print("Accuracy:", accuracy_score(y_test, preds))

    print("Saving model...")
    joblib.dump(model, model_dir / "logistic_model.pkl")

    print("Training complete.")


if __name__ == "__main__":
    main()
