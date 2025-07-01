import os
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# Assuming your original script is in the same directory or its path is accessible
# You might need to adjust this if your main script is in a different module
from train_with_vectorizer import MODEL_DIR, build_pipeline, truncate_float, REPORT_FILE

# Define the paths (should be consistent with your main script)
MODEL_DIR = "models"
REPORT_FILE = "report_test_set.csv" # Using a different report file for this test
PIPELINE_PATH = os.path.join(MODEL_DIR, "text_xgboost_pipeline.pkl")
LABELENC_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl") # Will be used if label encoding is needed

def create_test_data() -> pd.DataFrame:
    
    # nigerian_prince_emails = [
    #     "URGENT BUSINESS PROPOSAL: From Prince Aliko Dangote of Nigeria. I am writing to you today with an urgent and confidential business proposal. I am the son of the late King XYZ and due to political instability, a substantial amount of funds (US$25,000,000.00) has been frozen in an account. I require your assistance to transfer this fund to your account for a 20% commission. Please reply urgently with your bank details for more information.",
    #     "CONFIDENTIAL ASSISTANCE REQUIRED: Dear Esteemed Friend, I am Dr. John Doe, a high-ranking official with the Nigerian National Petroleum Corporation. I have discovered an over-invoiced contract sum of $15.5 Million USD. I need a reliable foreign partner to help us move this money out of the country. You will be compensated handsomely. Your immediate response is highly anticipated.",
    #     "ATTENTION: My Name is Barrister Mohammed, a legal counsel to a deceased client who shares your last name. He died without a next of kin and left behind a vast fortune of $10.5 million. I want to present you as the beneficiary. Please send your full name, address, and phone number for processing.",
    #     "URGENT FUND TRANSFER: Greetings, I am Prince Abdulaziz, heir to a prominent family. Due to unforeseen circumstances, I need your immediate assistance to transfer $30 Million USD from my family's dormant account. You will receive 30% of the total sum. Strict confidentiality is paramount. Reply to proceed.",
    #     "BUSINESS OPPORTUNITY: My dear sir/madam, I am Mr. George, an accountant at a bank here in Nigeria. A foreign client of ours passed away and left $8.7 million in his account. I seek your partnership to claim these funds, as there is no known beneficiary. Kindly provide your personal details for this mutually beneficial transaction.",
    #     "FROM THE DESK OF: Mr. David Williams, Central Bank of Nigeria. I wish to inform you of a security vault containing €18 million that needs to be transferred. We need a foreign account for this. You will be generously rewarded for your cooperation. Respond promptly for further instructions.",
    #     "PRIVATE INQUIRY: I am Mrs. Fatima, widow of a high-ranking government official. My late husband left a substantial amount, $12 million, in a security company. I am currently facing challenges accessing these funds and require a trustworthy individual like yourself to assist. Your assistance will not go unrewarded.",
    #     "SECURE TRANSFER NEEDED: Dear Friend, I am General Abu Bakar, seeking your help to move a sum of $22 million USD that is currently in a high-security diplomatic vault. I assure you of a significant percentage for your noble assistance. Time is of the essence.",
    #     "INHERITANCE NOTIFICATION: You have been selected as the beneficiary of an unclaimed inheritance from a distant relative, amounting to $9.5 million. To process this, we require a small legal fee to release the funds. Contact Barrister Ahmed for details.",
    #     "URGENT INVESTMENT PROPOSAL: I am writing to you on behalf of a consortium of businessmen in Nigeria who have a lucrative investment opportunity worth $35 million. We need a foreign partner to facilitate the transfer of funds. Your share will be substantial. Please indicate your interest."
    # ]

    nigerian_prince_emails = [ 
            "URGENT GRANT RELEASE: Dear Beneficiary, This is to inform you that your long-awaited humanitarian aid grant of $5,000,000.00 USD has been approved for release. Due to new international banking regulations, we require a small administrative fee of $500 to finalize the transfer. Please send the fee urgently via Bitcoin to facilitate immediate release of your funds. Act quickly, as this offer is time-sensitive.",
            "COVID-19 RELIEF FUND ALLOCATION: Greetings. As part of a global initiative to mitigate the economic impact of the recent pandemic, a substantial relief fund of $18.5 Million USD has been designated for you. To process your direct deposit, we require your bank account verification details and a clearance certificate fee. Respond immediately to avoid forfeiture.",
            "UNCLAIMED LOTTERY WINNINGS: Congratulations! Your email address was randomly selected as the lucky winner of €2,500,000.00 in the Euro Millions International Lottery. To claim your prize, you must first pay a tax and processing fee of €800. Kindly contact our claims agent, Dr. Obi, with your personal information and proof of payment.",
            "ASSISTANCE WITH ESTATE DISBURSEMENT: I am Barrister Kenneth, legal representative for a deceased client who passed away without a valid will. Extensive searches have revealed no next of kin. However, your last name matches that of the deceased. I propose we work together to claim his estate, valued at $17.2 million. Your prompt reply with your full details is crucial.",
            "LUCRATIVE INVESTMENT IN MINING: My name is Mr. Emeka, a senior official with a government parastatal overseeing large-scale gold mining operations. We have identified a highly profitable, untapped vein of gold worth $40 million. We seek a discreet foreign partner to help us secure the necessary equipment and facilitate export. A substantial percentage of the profits is guaranteed for your partnership. Confidentiality is key.",
            "URGENT BUSINESS TRANSACTION - OVER-INVOICED CONTRACT: I am contacting you on a matter of utmost urgency and confidentiality. As a senior auditor, I have uncovered an over-invoiced contract within the Ministry of Finance amounting to $13.8 Million USD. I require a reliable foreign account to transfer these funds. You will be generously compensated for your cooperation. Provide your banking information without delay.",
            "SECURITY VAULT RELEASE - DECEASED CLIENT FUNDS: My name is Mrs. Ngozi, a manager at a reputable security company. One of our foreign clients, now deceased, left a large sum of $9.1 million in a private vault. We have been unable to locate his next of kin. I propose a mutually beneficial arrangement where you pose as the legitimate beneficiary. Your immediate response will allow us to proceed.",
            "INTERNATIONAL COMPENSATION PAYOUT: This notification is from the United Nations Compensation Commission. Your name has been listed among those eligible for a compensation payout of $750,000.00 USD due to past financial injustices. To receive your funds, a processing and validation fee is required. Please follow the instructions in the attached document (which is a fake link/attachment).",
            "FROM THE DESK OF THE CENTRAL BANK GOVERNOR: We have identified a dormant account with a balance of $28.0 Million USD belonging to a foreign national who died intestate. As the Governor of the Central Bank, I require your urgent assistance to transfer these funds out of the country before they are confiscated. Your honesty and discretion are paramount. Reply with your full details for verification.",
            "SECURE DIPLOMATIC TRANSFER: I am General Sani, a high-ranking military officer. Due to recent political shifts, I need to urgently move $20 million USD currently held in a diplomatic account. I require a trusted foreign partner to receive these funds. Your share will be 25%. Time is of utmost importance for this sensitive transaction.",
            "OFFICIAL IMF PAYMENT RELEASE: Your pending payment of $1.5 Million USD from the International Monetary Fund (IMF) has been approved. However, a compulsory wire transfer charge of $350 is mandatory before release. This is a legitimate requirement for all international transfers. Send payment details promptly to avoid delay.",
            "DECEASED RELATIVE INHERITANCE: We have identified you as the last living relative of a distant, wealthy individual who passed away recently, leaving an inheritance of $11.3 million. To claim these funds, you will need to pay legal and probate fees. Contact our legal firm for a detailed breakdown of the required costs.",
            "CUSTOMS CLEARANCE ASSISTANCE - GOLD SHIPMENT: I am contacting you regarding a valuable gold shipment valued at $19 million, currently held by customs. Due to complex legalities, I require a foreign individual to act as the consignee. A small customs duty fee will be required from your end to release the shipment, and you will receive a percentage of the gold's value.",
            "URGENT BUSINESS COLLABORATION - GOVERNMENT CONTRACT: Our company has secured a significant government contract worth $32 million, but we face challenges in transferring funds due to strict local regulations. We are seeking a foreign business partner to facilitate the movement of these funds internationally. Your role will be handsomely rewarded with a 20% commission. Express your interest for a detailed proposal."
    ]



    legitimate_emails = [
        "Subject: Meeting Reminder - Project Alpha\n\nHi Team,\n\nJust a friendly reminder about our Project Alpha meeting scheduled for tomorrow, June 30, 2025, at 10:00 AM in Conference Room 3. Please come prepared to discuss the Q2 results. \n\nBest regards,\nSarah",
        "Subject: Your Order #12345 Has Shipped!\n\nDear Customer,\n\nGood news! Your recent order #12345 has been shipped and is expected to arrive within 3-5 business days. You can track your shipment here: [Tracking Link]\n\nThank you for your purchase!\n[Company Name]",
        "Subject: Quarterly Newsletter - June 2025\n\nHello Subscribers,\n\nCheck out our latest quarterly newsletter for updates on new features, product enhancements, and upcoming events. Read more here: [Newsletter Link]\n\nSincerely,\nThe Team at [Company Name]",
        "Subject: Invoice for Services Rendered - May 2025\n\nDear [Client Name],\n\nPlease find attached the invoice for services rendered in May 2025. The total amount due is $1,500.00, payable by July 15, 2025. Let me know if you have any questions.\n\nThanks,\nJohn",
        "Subject: Job Application Confirmation - Software Engineer\n\nDear [Applicant Name],\n\nThank you for your application for the Software Engineer position at [Company Name]. We have received your submission and will review it shortly. You will hear from us regarding the next steps within two weeks.\n\nRegards,\nHR Department",
        "Subject: Welcome to Our Community!\n\nHi [User Name],\n\nWelcome to [Community Name]! We're excited to have you join us. Get started by exploring our forums and connecting with other members. If you need any help, don't hesitate to reach out.\n\nWarmly,\nThe [Community Name] Team",
        "Subject: System Maintenance Notification\n\nDear User,\n\nThis is to inform you about scheduled system maintenance on July 5, 2025, from 1:00 AM to 3:00 AM GMT. During this period, some services may be temporarily unavailable. We apologize for any inconvenience.\n\nThank you for your understanding,\nIT Support",
        "Subject: Project Proposal Draft - Phase 1\n\nHi Alex,\n\nI've attached the draft for the Phase 1 project proposal. Please review it and provide your feedback by end of day tomorrow. Let's sync up if you have significant changes.\n\nBest,\nEmily",
        "Subject: Your Password Reset Request\n\nDear [User Name],\n\nWe received a request to reset your password. If you initiated this request, please click the link below to set a new password: [Reset Link]\n\nIf you did not request this, please ignore this email.\n\nSecurity Team,\n[Platform Name]",
        "Subject: Feedback Request: Recent Purchase Experience\n\nDear Valued Customer,\n\nWe hope you're enjoying your recent purchase from [Company Name]. We'd love to hear about your experience. Please take a few minutes to complete our survey: [Survey Link]\n\nYour feedback is important to us!\n[Company Name]"
    ]

    data = []
    for email in nigerian_prince_emails:
        data.append({"body": email, "label": 1})  # 1 for scam
    for email in legitimate_emails:
        data.append({"body": email, "label": 0})   # 0 for legitimate

    return pd.DataFrame(data)

def test_model():
    """
    Loads the trained pipeline and tests it with new sample emails.
    """
    if not os.path.exists(PIPELINE_PATH):
        print(f"Error: Model pipeline not found at {PIPELINE_PATH}.")
        print("Please run your main training script first to create the model.")
        return

    print("Loading trained pipeline...")
    with open(PIPELINE_PATH, 'rb') as f:
        pipeline = pickle.load(f)

    test_df = create_test_data()

    X_test_new = test_df.drop(columns=["label"])
    y_test_new = test_df["label"]

    print("\nRunning predictions on new test set...")

    t = pipeline.predict_email("testing if this is scam")

    print(t)
    print(" - -  - -  - -  - -  - -  - -  - -  - -  - -  - -  - - ")
    print(" - -  - -  - -  - -  - -  - -  - -  - -  - -  - -  - - ")
    print(" - -  - -  - -  - -  - -  - -  - -  - -  - -  - -  - - ")
    print(" - -  - -  - -  - -  - -  - -  - -  - -  - -  - -  - - ")


    y_pred = pipeline.predict(X_test_new)
    y_pred_prob = pipeline.predict_proba(X_test_new)[:, 1] # Probability of being scam (class 1)

    # Evaluate
    acc = accuracy_score(y_test_new, y_pred)
    prec = precision_score(y_test_new, y_pred)
    rec = recall_score(y_test_new, y_pred)
    f1 = f1_score(y_test_new, y_pred)
    auc = roc_auc_score(y_test_new, y_pred_prob)

    # Prepare and print summary report
    summary_report = pd.DataFrame({
        'ModelName': ['TFIDF+XGBoost (New Test Data)'],
        'Accuracy(%)': [f"{truncate_float(acc*100, 2)}"],
        'Precision(%)': [f"{truncate_float(prec*100, 2)}"],
        'Recall(%)': [f"{truncate_float(rec*100, 2)}"],
        'F1-Score(%)': [f"{truncate_float(f1*100, 2)}"],
        'ROC-AUC(%)': [f"{truncate_float(auc*100, 2)}"]
    })

    summary_report.to_csv(REPORT_FILE, index=False)
    print("Metrics report for new test set written to report_test_set.csv:\n")
    print(summary_report.to_string(index=False))

    # Prepare and print individual predictions table
    individual_predictions_data = []
    for i, (index, row) in enumerate(test_df.iterrows()):
        prediction = "SCAM" if y_pred[i] == 1 else "LEGIT"
        actual = "SCAM" if row['label'] == 1 else "LEGIT"
        email_snippet = row['body'].replace('\n', ' ') # Replace newlines for better table display
        if len(email_snippet) > 70: # Truncate for better table readability
            email_snippet = email_snippet[:67] + "..."
        individual_predictions_data.append({
            'Email #': i + 1,
            'Actual Label': actual,
            'Predicted Label': prediction,
            'Probability (SCAM)%': truncate_float(y_pred_prob[i] * 100, 2),
            'Email Body Snippet': email_snippet
        })

    individual_predictions_df = pd.DataFrame(individual_predictions_data)
    print("\nIndividual Email Predictions:\n")
    print(individual_predictions_df.to_string(index=False))

if __name__ == "__main__":
    test_model()
