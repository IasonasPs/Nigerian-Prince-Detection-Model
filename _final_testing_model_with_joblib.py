import os
import joblib  # Import joblib
from Entities.EmailSpamPipeline import EmailSpamPipeline 

MODEL_DIR = "ML_Models"
PIPELINE_PATH_JOBLIB = os.path.join(MODEL_DIR, "text_xgboost_pipeline.joblib") # Path for joblib model

def load_pipeline_joblib() -> EmailSpamPipeline:
    """
    Loads the EmailSpamPipeline from a joblib file.
    """
    return joblib.load(PIPELINE_PATH_JOBLIB)

def predict_email_joblib(email_body: str) -> dict:
    """
    Uses the joblib-loaded pipeline to predict if an email is spam.

    Args:
        email_body (str): The content of the email to predict.

    Returns:
        dict: A dictionary containing the prediction (True for spam, False for not spam)
              and the probability of it being spam.
    """
    pipeline = load_pipeline_joblib()
    return pipeline.predict_email(email_body)

# Example use
if __name__ == "__main__":
    # Ensure a .joblib model exists in the ML_Models directory for this to run correctly.
    # You would typically run the training script first to save the model.

    print("--- Using Joblib Loaded Model ---")
    result_joblib = predict_email_joblib("URGENT GRANT RELEASE: Dear Beneficiary, This is to inform you that your long-awaited humanitarian aid grant of $5,000,000.00 USD has been approved for release. Due to new international banking regulations, we require a small administrative fee of $500 to finalize the transfer. Please send the fee urgently via Bitcoin to facilitate immediate release of your funds. Act quickly, as this offer is time-sensitive.")
    
    if (result_joblib.prediction):
        print(f"This email is likely a scam with a probability of {result_joblib.probability:.2f}.")
    else:
        print(f"This email is likely not a scam with a probability of {result_joblib.probability:.2f}.") 
    
    print("\n--- Example with a non-spam email ---")
    result_non_spam = predict_email_joblib("Hi John, just a reminder about our meeting tomorrow at 10 AM. Please bring the quarterly report. Best, Mary.")
    if (result_non_spam.prediction):
        print(f"This email is likely a scam with a probability of {result_non_spam.probability:.2f}.")
    else:
        print(f"This email is likely not a scam with a probability of {result_non_spam.probability:.2f}.")