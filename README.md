# 📰 Fake News Classification – NLP Project

![Ironhack Logo](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

## 📌 Project Overview
This project applies **Natural Language Processing (NLP)** and **Machine Learning** to classify news articles as **REAL (0)** or **FAKE (1)**.  

We compare different models (Logistic Regression, Naive Bayes, Random Forest, Linear SVC, Word2Vec embeddings) and evaluate their performance. The final best-performing model is deployed and tested.

---

## 📂 Dataset
The dataset (`dataset/data.csv`) contains ~40,000 news articles with:
- `label`: 0 = fake, 1 = real  
- `title`: headline  
- `text`: article body  
- `subject`: topic/category  
- `date`: publication date  

👉 For predictions, we use `dataset/validation_data.csv` where labels are `2` and must be replaced by model predictions (`0` or `1`).

---

## ⚙️ Approach
1. **Data Cleaning & Preprocessing**
   - Removed `subject` and `date` columns (irrelevant for classification).
   - Combined `title` and `text`.
   - Tokenization, stopword removal, lowercasing.
2. **Feature Engineering**
   - TF-IDF vectorization.
   - Word2Vec embeddings (tested separately).
3. **Model Training**
   - Logistic Regression, Naive Bayes, Random Forest, Linear SVC.
   - Hyperparameter tuning, class balance checked.
4. **Evaluation**
   - Accuracy, Precision, Recall, F1.
   - Confusion matrices and metrics comparison.
5. **Deployment**
   - Streamlit app for interactive demo.
   - Predictions for `validation_data.csv`.

---

## 🏆 Results
- **Linear SVC + TF-IDF** performed best:
  - Accuracy: **99.42%**
  - Precision: **99.40%**
  - Recall: **99.45%**
  - F1: **99.43%**
- Logistic Regression also strong (98.45%).
- Word2Vec embeddings underperformed TF-IDF slightly.

---

## 📊 Deliverables
1. **Code** → training & evaluation notebooks/scripts.  
2. **Predictions** → `validation_predictions.csv` (labels replaced with model outputs).  
3. **Accuracy Estimation** → reported test metrics (see Results).  
4. **Presentation** → PowerPoint/PDF slides summarizing methodology & results.  
5. **Streamlit App** → simple web interface to test the model.
