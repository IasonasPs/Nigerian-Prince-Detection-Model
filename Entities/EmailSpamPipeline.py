import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

from Entities.PredictionResult import PredictionResult

class EmailSpamPipeline(Pipeline):

    def __init__(self):
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

        preprocessor = ColumnTransformer([
            ("body_txt", text_pipeline, "body")
        ], remainder="drop")

        super().__init__([
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(
                eval_metric='logloss',
                use_label_encoder=False,
                random_state=42
            ))
        ])

    def predict_email(self, email_body: str) -> PredictionResult:
        df = pd.DataFrame({"body": [email_body]})
        
        result = PredictionResult(
            prediction=bool(self.predict(df)[0]),
            probability=self.predict_proba(df)[0, 1]
        )
        
        return result
