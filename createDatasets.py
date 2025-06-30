import pandas as pd
from sklearn.model_selection import train_test_split
import os

df = pd.read_csv("combined_emails.csv", encoding="utf-8")

valid_labels = set([0, 1])
labels_in_data = set(df['label'].unique())
if not labels_in_data <= valid_labels:
    raise ValueError(f"Label column contains unexpected values: {labels_in_data - valid_labels}")

#Split into training and test sets (80/20)
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['label'] 
)

# Save to CSV
if not os.path.exists("datasets"):
    os.makedirs("datasets")

train_df.to_csv("datasets/train.csv", index=False, encoding="utf-8")
test_df.to_csv("datasets/test.csv", index=False, encoding="utf-8")

print(f"Training set saved as 'train.csv' ({len(train_df)} rows)")
print(f"Test set saved as 'test.csv' ({len(test_df)} rows)")