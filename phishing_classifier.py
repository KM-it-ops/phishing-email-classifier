"""
Phishing Email Classifier
==========================
A machine learning pipeline that classifies emails as phishing or legitimate
using text-based feature extraction and a Random Forest classifier.

Author: Michael Kurdi
Project: Portfolio — Cybersecurity + AI
Tools: Python, scikit-learn, pandas, NumPy
"""

import os
import json
import csv
import re
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# ---------------------------------------------------------------------------
# 1. DATASET GENERATION
# ---------------------------------------------------------------------------
# In a production environment you would use a real corpus (e.g., the Nazario
# phishing corpus or the IWSPA dataset). For this portfolio project we
# generate a small, representative synthetic dataset so the project is
# fully self-contained and reproducible without external downloads.
# ---------------------------------------------------------------------------

PHISHING_TEMPLATES = [
    "URGENT: Your account has been compromised. Click here immediately to verify your identity: {link}",
    "Dear Customer, We detected unusual activity on your account. Please confirm your credentials at {link}",
    "You have won a $1,000 gift card! Claim your prize now by visiting {link}",
    "Your payment of $499.99 has been processed. If you did not authorize this, click {link}",
    "ACTION REQUIRED: Your password expires in 24 hours. Reset it now at {link}",
    "Hi, I am a prince from Nigeria and I need your help transferring $5,000,000. Reply with your bank details.",
    "Security Alert: Someone tried to sign in to your account. Verify your identity: {link}",
    "Congratulations! You've been selected for an exclusive offer. Act now before it expires: {link}",
    "Your Apple ID has been locked. To unlock, verify your information at {link}",
    "IMPORTANT: IRS refund of $3,247.00 is pending. Submit your SSN and bank info at {link}",
    "Dear user, your Netflix subscription will be cancelled unless you update payment at {link}",
    "Warning: Virus detected on your computer. Download our free scanner at {link}",
    "Your Amazon order #938-284719 cannot be delivered. Update shipping info: {link}",
    "Bank of America: Suspicious login detected. Secure your account immediately: {link}",
    "You have (1) unread message from a friend. View it here: {link}",
    "PayPal: We noticed irregular activity. Confirm your identity to avoid suspension: {link}",
    "Final Notice: Your account will be permanently deleted in 48 hours. Verify now: {link}",
    "FREE iPhone 15 giveaway! You are one of 10 lucky winners. Claim yours: {link}",
    "Microsoft Office 365: Your license key is expired. Renew immediately at {link}",
    "Dear valued customer, please verify your social security number for tax purposes at {link}",
]

LEGITIMATE_TEMPLATES = [
    "Hi team, please find attached the Q3 performance report. Let me know if you have questions.",
    "Meeting rescheduled to Thursday at 2 PM. Updated calendar invite sent.",
    "Thanks for your order! Your tracking number is 1Z999AA10123456784. Estimated delivery: Friday.",
    "Reminder: Project deadline is next Monday. Please submit your sections by EOD Friday.",
    "Welcome to our newsletter! Here are this week's top articles on cybersecurity trends.",
    "Your monthly bank statement for October is now available in your online banking portal.",
    "Hi, just following up on our conversation from last week. Do you have time for a call tomorrow?",
    "The board meeting minutes from Tuesday have been uploaded to the shared drive.",
    "Your subscription renewal was successful. Next billing date: March 15, 2026.",
    "Please review the attached contract and provide your feedback by Wednesday.",
    "Team standup notes: Sprint 4 retrospective scheduled for Friday at 10 AM.",
    "Your flight confirmation: AA 1742, departing CLT 6:45 AM, arriving JFK 9:10 AM.",
    "Congratulations on completing the training module! Your certificate is attached.",
    "IT Maintenance Notice: Systems will be down Saturday 2-6 AM for scheduled updates.",
    "Here is the updated project timeline. Key milestones are highlighted in yellow.",
    "Your prescription is ready for pickup at the pharmacy. Store hours: 9 AM - 9 PM.",
    "Thank you for attending the webinar. Recording and slides are available at the link below.",
    "Expense report approved. Reimbursement of $342.17 will appear in your next paycheck.",
    "Please complete your annual compliance training by December 31. Access it through the HR portal.",
    "Happy birthday from the team! Hope you have a great day.",
]

PHISHING_LINKS = [
    "http://secure-verify-account.com/login",
    "http://bit.ly/3xFkZ9q",
    "http://192.168.1.100/verify",
    "http://amaz0n-security.net/confirm",
    "http://paypa1-secure.com/update",
    "http://micr0soft-365.com/renew",
    "http://login-bankofamerica.xyz/auth",
]


def generate_dataset(n_phishing: int = 200, n_legitimate: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic email dataset for classification."""
    rng = np.random.default_rng(seed)
    emails = []

    for _ in range(n_phishing):
        template = rng.choice(PHISHING_TEMPLATES)
        link = rng.choice(PHISHING_LINKS)
        body = template.format(link=link) if "{link}" in template else template
        # Add random noise: typos, extra urgency words
        if rng.random() < 0.3:
            body = body.upper()
        if rng.random() < 0.4:
            body += " Act now! Limited time!"
        emails.append({"text": body, "label": 1})  # 1 = phishing

    for _ in range(n_legitimate):
        template = rng.choice(LEGITIMATE_TEMPLATES)
        # Slight variation
        if rng.random() < 0.2:
            template = "FYI - " + template
        if rng.random() < 0.15:
            template += " Best regards, Management."
        emails.append({"text": template, "label": 0})  # 0 = legitimate

    df = pd.DataFrame(emails)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------------------------

class EmailFeatureExtractor:
    """Extract security-relevant features from email text."""

    URGENCY_WORDS = {
        "urgent", "immediately", "action required", "act now", "expires",
        "suspended", "locked", "compromised", "verify", "confirm",
        "warning", "alert", "final notice", "limited time",
    }

    SUSPICIOUS_PATTERNS = [
        r"http[s]?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",  # IP-based URLs
        r"http[s]?://[a-z0-9-]+\.(xyz|tk|ml|ga|cf|gq)",     # Suspicious TLDs
        r"\b(ssn|social\s*security|bank\s*details?|password)\b",  # PII requests
        r"[a-z0-9]+[01][a-z0-9]+-",                          # Typosquatted domains (e.g., amaz0n)
    ]

    def extract(self, text: str) -> dict:
        """Extract hand-crafted features from a single email."""
        text_lower = text.lower()
        words = text_lower.split()

        features = {}

        # Urgency score: count of urgency keywords
        features["urgency_score"] = sum(
            1 for kw in self.URGENCY_WORDS if kw in text_lower
        )

        # Suspicious pattern count
        features["suspicious_patterns"] = sum(
            1 for pat in self.SUSPICIOUS_PATTERNS if re.search(pat, text_lower)
        )

        # URL count
        features["url_count"] = len(re.findall(r"http[s]?://\S+", text))

        # Ratio of uppercase characters (phishing often uses ALL CAPS)
        alpha_chars = [c for c in text if c.isalpha()]
        features["caps_ratio"] = (
            sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
            if alpha_chars
            else 0
        )

        # Exclamation mark count
        features["exclamation_count"] = text.count("!")

        # Text length
        features["text_length"] = len(text)

        # Contains monetary reference
        features["has_money_ref"] = int(bool(re.search(r"\$[\d,]+\.?\d*", text)))

        return features

    def extract_dataframe(self, df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
        """Extract features for an entire DataFrame."""
        feature_rows = [self.extract(row) for row in df[text_col]]
        return pd.DataFrame(feature_rows)


# ---------------------------------------------------------------------------
# 3. MODEL TRAINING & EVALUATION
# ---------------------------------------------------------------------------

def train_and_evaluate(df: pd.DataFrame, output_dir: str = "results"):
    """Full training pipeline with TF-IDF + hand-crafted features."""
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("PHISHING EMAIL CLASSIFIER — TRAINING PIPELINE")
    print("=" * 60)

    # --- Feature extraction ---
    print("\n[1/5] Extracting features...")
    extractor = EmailFeatureExtractor()
    hand_crafted = extractor.extract_dataframe(df)

    # TF-IDF on email text
    tfidf = TfidfVectorizer(max_features=200, stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(df["text"])
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(), columns=[f"tfidf_{w}" for w in tfidf.get_feature_names_out()]
    )

    # Combine features
    X = pd.concat([hand_crafted, tfidf_df], axis=1)
    y = df["label"]

    print(f"  Total features: {X.shape[1]}")
    print(f"  Hand-crafted:   {hand_crafted.shape[1]}")
    print(f"  TF-IDF:         {tfidf_df.shape[1]}")
    print(f"  Samples:        {len(df)} (Phishing: {sum(y)}, Legitimate: {len(y) - sum(y)})")

    # --- Train/test split ---
    print("\n[2/5] Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Model training ---
    print("\n[3/5] Training Random Forest classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    # --- Cross-validation ---
    print("\n[4/5] Running 5-fold cross-validation...")
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="f1")
    print(f"  CV F1 scores: {[f'{s:.3f}' for s in cv_scores]}")
    print(f"  Mean CV F1:   {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    # --- Test evaluation ---
    print("\n[5/5] Evaluating on test set...")
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n  Accuracy:  {accuracy:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1 Score:  {f1:.3f}")

    print(f"\n  Classification Report:")
    report = classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"])
    print(report)

    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion Matrix:")
    print(f"                  Predicted Legit  Predicted Phish")
    print(f"  Actual Legit    {cm[0][0]:>15}  {cm[0][1]:>15}")
    print(f"  Actual Phish    {cm[1][0]:>15}  {cm[1][1]:>15}")

    # --- Feature importance ---
    importances = clf.feature_importances_
    feature_names = X.columns.tolist()
    top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:15]

    print(f"\n  Top 15 Features by Importance:")
    for name, imp in top_features:
        bar = "#" * int(imp * 100)
        print(f"    {name:<30} {imp:.4f}  {bar}")

    # --- Save results ---
    results = {
        "dataset_size": len(df),
        "phishing_count": int(sum(y)),
        "legitimate_count": int(len(y) - sum(y)),
        "total_features": int(X.shape[1]),
        "test_accuracy": round(accuracy, 4),
        "test_precision": round(precision, 4),
        "test_recall": round(recall, 4),
        "test_f1": round(f1, 4),
        "cv_f1_mean": round(cv_scores.mean(), 4),
        "cv_f1_std": round(cv_scores.std(), 4),
        "confusion_matrix": cm.tolist(),
        "top_features": [{"name": n, "importance": round(i, 4)} for n, i in top_features],
    }

    with open(os.path.join(output_dir, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to {output_dir}/evaluation_results.json")

    return clf, results


# ---------------------------------------------------------------------------
# 4. DEMO: CLASSIFY NEW EMAILS
# ---------------------------------------------------------------------------

def demo_classify(clf, tfidf, extractor, emails: list[str]):
    """Demonstrate classification on new, unseen emails."""
    print("\n" + "=" * 60)
    print("DEMO: CLASSIFYING NEW EMAILS")
    print("=" * 60)

    for i, email in enumerate(emails, 1):
        hand_features = pd.DataFrame([extractor.extract(email)])
        tfidf_features = pd.DataFrame(
            tfidf.transform([email]).toarray(),
            columns=[f"tfidf_{w}" for w in tfidf.get_feature_names_out()],
        )
        features = pd.concat([hand_features, tfidf_features], axis=1)
        prediction = clf.predict(features)[0]
        proba = clf.predict_proba(features)[0]

        label = "PHISHING" if prediction == 1 else "LEGITIMATE"
        confidence = max(proba) * 100

        print(f"\n  Email {i}: \"{email[:80]}{'...' if len(email) > 80 else ''}\"")
        print(f"  Result:     {label}")
        print(f"  Confidence: {confidence:.1f}%")
        print(f"  Phishing probability: {proba[1]:.3f}")


# ---------------------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------------------

def main():
    # Generate dataset
    print("Generating synthetic email dataset...")
    df = generate_dataset(n_phishing=200, n_legitimate=200)

    # Save dataset for reproducibility
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/email_dataset.csv", index=False)
    print(f"Dataset saved to data/email_dataset.csv ({len(df)} emails)\n")

    # Train and evaluate
    clf, results = train_and_evaluate(df)

    # Re-fit TF-IDF on full training text for demo (same as training pipeline)
    tfidf = TfidfVectorizer(max_features=200, stop_words="english", ngram_range=(1, 2))
    tfidf.fit(df["text"])
    extractor = EmailFeatureExtractor()

    # Demo with new emails
    test_emails = [
        "URGENT: Your bank account has been suspended! Click http://192.168.1.50/verify to restore access immediately!",
        "Hi Michael, the team lunch is scheduled for Friday at noon. Conference room B. See you there!",
        "You have been selected to receive a FREE $500 Walmart gift card! Claim now: http://fr33-gift.xyz/claim",
        "Please review the attached quarterly report and send your comments by Thursday.",
        "WARNING: Your Netflix account will be cancelled in 24 hours unless you update payment info at http://netf1ix-billing.com/update",
    ]
    demo_classify(clf, tfidf, extractor, test_emails)

    print("\n" + "=" * 60)
    print("Pipeline complete. All results saved to results/ directory.")
    print("=" * 60)


if __name__ == "__main__":
    main()
