import os
import shutil
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from Entities.EmailSpamPipeline import EmailSpamPipeline  

# Constants
MODEL_DIR = "ML_Models"
REPORT_FILE = "Report.csv"
TRAIN_PATH = os.path.join("Datasets", "train.csv")
TEST_PATH = os.path.join("Datasets", "test.csv")
PIPELINE_PATH = os.path.join(MODEL_DIR, "text_xgboost_pipeline.pkl")

def truncate_float(number: float, decimals: int) -> float:
    factor = 10 ** decimals
    return int(number * factor) / factor

def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8")

def save_pipeline(pipeline: EmailSpamPipeline):
    with open(PIPELINE_PATH, "wb") as f:
        pickle.dump(pipeline, f)

def load_pipeline() -> EmailSpamPipeline:
    with open(PIPELINE_PATH, "rb") as f:
        return pickle.load(f)

def evaluate(pipeline: EmailSpamPipeline, X_test: pd.DataFrame, y_test: pd.Series):
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    return {
        "Accuracy(%)": truncate_float(acc * 100, 2),
        "Precision(%)": truncate_float(prec * 100, 2),
        "Recall(%)": truncate_float(rec * 100, 2),
        "F1-Score(%)": truncate_float(f1 * 100, 2),
        "ROC-AUC(%)": truncate_float(auc * 100, 2)
    }

def write_report(metrics: dict):
    df = pd.DataFrame([{
        "ModelName": "TFIDF+XGBoost",
        **metrics
    }])
    df.to_csv(REPORT_FILE, index=False)
    print("📄 Metrics report written to report.csv:\n")
    print(df.to_string(index=False))

def main():
    if os.path.exists(MODEL_DIR):
        shutil.rmtree(MODEL_DIR)
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(REPORT_FILE):
        os.remove(REPORT_FILE)

    # Load and clean data
    train_df = load_dataset(TRAIN_PATH)
    test_df = load_dataset(TEST_PATH)
    train_df.columns = train_df.columns.str.strip()
    test_df.columns = test_df.columns.str.strip()

    X_train, y_train = train_df.drop(columns=["label"]), train_df["label"]
    X_test, y_test = test_df.drop(columns=["label"]), test_df["label"]

    print("\n🚀 Training pipeline...")
    pipeline = EmailSpamPipeline()
    pipeline.fit(X_train, y_train)

    save_pipeline(pipeline)
    print(f" Pipeline saved to {PIPELINE_PATH}")

    print("\n Evaluating model...")
    metrics = evaluate(pipeline, X_test, y_test)
    write_report(metrics)

if __name__ == "__main__":
    main()
