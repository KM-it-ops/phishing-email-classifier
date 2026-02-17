# Phishing Email Classifier

A machine learning pipeline that classifies emails as **phishing** or **legitimate** using text-based feature extraction and a Random Forest classifier.

## Goal

Demonstrate the ability to build an end-to-end ML pipeline for a cybersecurity use case: detecting phishing emails through feature engineering, model training, and evaluation.

## Tools & Technologies

- **Python 3.10+**
- **scikit-learn** — TF-IDF vectorization, Random Forest classifier, cross-validation
- **pandas / NumPy** — Data manipulation and numerical operations
- **Re (regex)** — Pattern matching for suspicious URLs, PII requests, typosquatting

## Architecture

```
Email Text
    │
    ├──► Hand-Crafted Features (7 features)
    │       • Urgency keyword score
    │       • Suspicious URL pattern count
    │       • URL count
    │       • Uppercase character ratio
    │       • Exclamation mark count
    │       • Text length
    │       • Monetary reference flag
    │
    ├──► TF-IDF Features (200 features)
    │       • Unigrams and bigrams
    │       • English stop words removed
    │
    └──► Combined Feature Vector (207 features)
              │
              ▼
        Random Forest Classifier (100 trees)
              │
              ▼
        Prediction: PHISHING or LEGITIMATE
        + Confidence Score
```

## How to Run

```bash
# Install dependencies
pip install scikit-learn pandas numpy

# Run the full pipeline
python phishing_classifier.py
```

This will:
1. Generate a synthetic dataset of 400 emails (200 phishing, 200 legitimate)
2. Extract hand-crafted + TF-IDF features
3. Train a Random Forest classifier with 80/20 train/test split
4. Run 5-fold cross-validation
5. Print evaluation metrics and top features
6. Demonstrate classification on 5 new unseen emails
7. Save results to `results/evaluation_results.json`

## Results

| Metric    | Score |
|-----------|-------|
| Accuracy  | 100%  |
| Precision | 100%  |
| Recall    | 100%  |
| F1 Score  | 100%  |

**Cross-Validation (5-fold):**
- Mean F1: **99.68%**
- Std Dev: ±0.63%

**Confusion Matrix (Test Set):**
```
                  Predicted Legit  Predicted Phish
Actual Legit                   40                0
Actual Phish                    0               40
```
## Visualizations

### Performance Metrics
![Performance Metrics](screenshots/performance_metrics.png)

### Confusion Matrix
![Confusion Matrix](screenshots/confusion_matrix.png)

### Top Detection Features
![Feature Importance](screenshots/feature_importance.png)

### Cross-Validation Stability
![Cross-Validation Results](screenshots/cross_validation.png)

### Dataset Composition
![Dataset Distribution](screenshots/dataset_distribution.png)

[View full evaluation metrics](results/evaluation_results.json)

## Key Features Detected

The classifier learns to identify phishing signals including:

| Feature | Importance | Description |
|---------|------------|-------------|
| `http` (TF-IDF) | 17.35% | Presence of URLs in email body |
| Suspicious patterns | 8.36% | IP-based URLs, typosquatted domains, PII requests |
| Text length | 6.58% | Phishing emails tend to be shorter/more urgent |
| Urgency score | 6.35% | Keywords like "immediately", "act now", "expires" |
| URL count | 6.35% | Multiple links often indicate phishing |

Top detection signals: URL presence, urgency language, suspicious domain patterns, and ALL CAPS formatting.

## Limitations

- **Synthetic dataset**: Uses generated emails rather than a real-world corpus. Production deployment would require training on actual phishing/legitimate email datasets (e.g., Nazario corpus, IWSPA).
- **No header analysis**: Real phishing detection also examines email headers (SPF, DKIM, sender reputation). This project focuses on body text only.
- **Limited adversarial robustness**: Sophisticated phishing that mimics legitimate tone may evade detection. Adversarial testing was not performed.
- **Class balance assumption**: The 50/50 split does not reflect real-world distribution where phishing is a smaller percentage of total email volume.
- **No deployment pipeline**: This is a batch analysis tool, not a real-time email filter.

## Lessons Learned

1. **Feature engineering matters**: Hand-crafted features (urgency score, suspicious URL patterns) provided strong signal alongside TF-IDF, demonstrating that domain knowledge improves ML model performance.
2. **Evaluation beyond accuracy**: Using precision, recall, and F1 gives a more complete picture — in phishing detection, recall (catching all phishing) often matters more than precision.
3. **Documentation is part of the work**: Clearly documenting limitations and assumptions demonstrates professional maturity and honest technical assessment.

## Author

Michael Kurdi — [LinkedIn](https://www.linkedin.com/in/michael-kurdi) | CompTIA Security+ | B.S. Information Technology (Cybersecurity), SNHU
