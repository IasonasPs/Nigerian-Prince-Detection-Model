import os
import pickle
from Entities.EmailSpamPipeline import EmailSpamPipeline  

MODEL_DIR = "ML_Models"
PIPELINE_PATH = os.path.join(MODEL_DIR, "text_xgboost_pipeline.pkl")

def load_pipeline() -> EmailSpamPipeline:
    with open(PIPELINE_PATH, "rb") as f:
        return pickle.load(f)

def predict_email(email_body: str) -> dict:
    pipeline = load_pipeline()
    return pipeline.predict_email(email_body)

# Example use
if __name__ == "__main__":
    result = predict_email("URGENT GRANT RELEASE: Dear Beneficiary, This is to inform you that your long-awaited humanitarian aid grant of $5,000,000.00 USD has been approved for release. Due to new international banking regulations, we require a small administrative fee of $500 to finalize the transfer. Please send the fee urgently via Bitcoin to facilitate immediate release of your funds. Act quickly, as this offer is time-sensitive.")
    
    if (result.prediction):
        print(f"This email is likely a scam with a probability of {result.probability:.2f}.")
    else:
        print(f"This email is likely not a scam with a probability of {result.probability:.2f}.")    
    