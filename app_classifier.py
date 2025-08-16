
import streamlit as st
import joblib

# Load your trained model (Pipeline with TF-IDF + Classifier)
model = joblib.load("linear_svc_model.pkl")

st.title("📰 Fake News Classifier")
st.write("Paste a news article below and see if it's REAL or FAKE.")

# User input
title = st.text_input("Title (optional)")
text  = st.text_area("Content", height=200)

if st.button("Classify"):
    doc = (title + " " + text).strip()
    if not doc:
        st.warning("Please enter a title and/or content.")
    else:
        # Try probability first (LogReg/NaiveBayes/Calibrated SVC), else fallback to margin (LinearSVC)
        label = model.predict([doc])[0]  # 0=REAL, 1=FAKE
        try:
            proba = model.predict_proba([doc])[0, 1]  # P(fake)
            conf_txt = f"Confidence (P(fake)): {proba:.2f}"
        except Exception:
            # Decision function to a friendly number (not a true probability)
            try:
                m = float(model.decision_function([doc])[0])
                proba_like = 1 / (1 + np.exp(-m))  # simple squash for display
                conf_txt = f"Margin score: {m:.2f}  (≈{proba_like:.2f})"
            except Exception:
                conf_txt = "Confidence not available."

        if label == 1:
            st.error("🚨 FAKE")
        else:
            st.success("✅ REAL")
        st.write(conf_txt)