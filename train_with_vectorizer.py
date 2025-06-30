import os
import pickle
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

MODEL_DIR = "models"
REPORT_FILE = "report.csv"
TRAIN_PATH = os.path.join("datasets", "train.csv")
TEST_PATH = os.path.join("datasets", "test.csv")


def truncate_float(number: float, decimals: int) -> float:
    """
    Truncates a float to a specified number of decimal places.
    """
    factor = 10 ** decimals
    return int(number * factor) / factor


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8")


def build_pipeline() -> Pipeline:
    # TF-IDF sub-pipeline for text
    text_pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                stop_words='english',
                max_df=0.9,
                min_df=5,
                ngram_range=(1, 2)
            )
        )
    ])

    # Combine transformers
    preprocessor = ColumnTransformer([
        ("body_txt", text_pipeline, "body")
    ], remainder="drop")

    # Full pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        (
            "classifier",
            XGBClassifier(
                eval_metric='logloss',
                use_label_encoder=False,
                random_state=42
            )
        )
    ])

    return pipeline


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(REPORT_FILE):
        os.remove(REPORT_FILE)

    train_df = load_dataset(TRAIN_PATH)
    test_df = load_dataset(TEST_PATH)

    train_df.columns = train_df.columns.str.strip()
    test_df.columns = test_df.columns.str.strip()

    X_train, y_train = train_df.drop(columns=["label"]), train_df["label"]
    X_test,  y_test  = test_df.drop(columns=["label"]),  test_df["label"]

    # Encode target labels
    y_train_enc = y_train
    y_test_enc  = y_test

    print("\nBuilding and training pipeline...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train_enc)

    # Save pipeline and label encoder
    pipeline_path = os.path.join(MODEL_DIR, "text_xgboost_pipeline.pkl")
    labelenc_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
    with open(pipeline_path, 'wb') as f:
        pickle.dump(pipeline, f)
    # with open(labelenc_path, 'wb') as f:
    #     pickle.dump(le_label, f)

    print(f"Saved pipeline to {pipeline_path}")
    print(f"Saved label encoder to {labelenc_path}\n")

    # Predict and evaluate
    print("\nRunning predictions on test set...")
    y_pred     = pipeline.predict(X_test)
    y_pred_prob = pipeline.predict_proba(X_test)[:, 1]

    acc   = accuracy_score(y_test_enc, y_pred)
    prec  = precision_score(y_test_enc, y_pred)
    rec   = recall_score(y_test_enc, y_pred)
    f1    = f1_score(y_test_enc, y_pred)
    auc   = roc_auc_score(y_test_enc, y_pred_prob)

    # Prepare report
    report = pd.DataFrame({
        'ModelName': ['TFIDF+XGBoost'],
        'Accuracy(%)': [f"{truncate_float(acc*100, 2)}"],
        'Precision(%)': [f"{truncate_float(prec*100, 2)}"],
        'Recall(%)': [f"{truncate_float(rec*100, 2)}"],
        'F1-Score(%)': [f"{truncate_float(f1*100, 2)}"],
        'ROC-AUC(%)': [f"{truncate_float(auc*100, 2)}"]
    })

    report.to_csv(REPORT_FILE, index=False)
    print("Metrics report written to report.csv:\n")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
