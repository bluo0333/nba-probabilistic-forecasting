from pathlib import Path
import duckdb
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss


def main():
    repo_root = Path(__file__).resolve().parents[1]
    db_path = repo_root / "data" / "nba.duckdb"
    model_dir = repo_root / "models"
    model_dir.mkdir(exist_ok=True)

    conn = duckdb.connect(str(db_path))

    print("Loading features...")
    df = conn.execute("SELECT * FROM final_features").df()

    train = df[df["game_date"] < "2018-01-01"]
    test = df[df["game_date"] >= "2018-01-01"]

    features = [
        "net_diff_last5",
        "win_pct_diff_last5",
        "elo_diff",
        "rest_diff",
        "b2b_diff",
        "netrtg_diff_last10"
    ]

    X_train = train[features]
    y_train = train["home_win"]

    X_test = test[features]
    y_test = test["home_win"]

    print("Training model...")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logit", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]

    print("Log Loss:", log_loss(y_test, probs))
    print("Brier Score:", brier_score_loss(y_test, probs))

    print("Saving model...")
    joblib.dump(model, model_dir / "logistic_model.pkl")

    print("Training complete.")


if __name__ == "__main__":
    main()
