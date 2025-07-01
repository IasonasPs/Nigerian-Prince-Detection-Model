import os
import pickle
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from types import MethodType
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import shutil
import Entities.EmailSpamPipeline as EmailSpamPipeline


MODEL_DIR = "models"
REPORT_FILE = "report.csv"
TRAIN_PATH = os.path.join("datasets", "train.csv")
TEST_PATH = os.path.join("datasets", "test.csv")


def truncate_float(number: float, decimals: int) -> float:
    # Truncates a float to a specified number of decimal places.
    
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

def predict_email(email_body: str) -> dict:
        """
        Predicts whether the given email body is spam or not using the trained pipeline.
        Returns a dictionary with prediction and probability.
        """
        pipeline_path = os.path.join(MODEL_DIR, "text_xgboost_pipeline.pkl")
        if not os.path.exists(pipeline_path):
            raise FileNotFoundError("Trained pipeline not found. Please train the model first.")

        with open(pipeline_path, "rb") as f:
            pipeline = pickle.load(f)

        input_df = pd.DataFrame({"body": [email_body]})
        pred = pipeline.predict(input_df)[0]
        prob = pipeline.predict_proba(input_df)[0, 1]

        return {"prediction": bool(pred), "probability": float(prob)}

def add_predict_email_to_pipeline(pipeline: Pipeline):
    # Adds a 'predict_email' method to the pipeline instance for single email prediction.
    print("Adding 'predict_email' method to pipeline...")
    def predict_email(self, email_body: str) -> dict:
        input_df = pd.DataFrame({"body": [email_body]})
        pred = self.predict(input_df)[0]
        prob = self.predict_proba(input_df)[0, 1]
        return {"prediction": bool(pred), "probability": float(prob)}
    # Bind the method to the pipeline instance
    pipeline.predict_email = MethodType(predict_email, pipeline)

      
def main():
    # os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(MODEL_DIR):
        shutil.rmtree(MODEL_DIR)
    os.makedirs(MODEL_DIR)



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

    add_predict_email_to_pipeline(pipeline)
    pipeline = EmailSpamPipeline(steps=[...])


    
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
