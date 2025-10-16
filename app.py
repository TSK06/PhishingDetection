# app.py
import streamlit as st
from lottie_streamlit import load_lottieurl
from prediction import predictor, classifier
from preprocess import cvectorizer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from dotenv import load_dotenv
load_dotenv()



##Groq
from groq import Groq
import os

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def chatbot_response(user_input):
    response = client.chat.completions.create(
        model= "llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful assistant specialized in phishing detection and cybersecurity. Keep answers short and clear."},
            {"role": "user", "content": user_input},
        ],
    )
    return response.choices[0].message.content



# If classifier is GridSearchCV, unwrap it
model = getattr(classifier, "best_estimator_", classifier)

# Initialize session state
if "total" not in st.session_state:
    st.session_state.total = 0
    st.session_state.safe = 0
    st.session_state.malicious = 0
    st.session_state.history = []

def explain_prediction(url, label):
    """Explain prediction for linear SVM using coefficients"""
    st.subheader("Why this prediction?")
    vectorized = cvectorizer.transform([url])

    if hasattr(model, "coef_"):
        feature_names = cvectorizer.get_feature_names_out()
        coefs = model.coef_.toarray()[0]

        indices = vectorized.nonzero()[1]
        token_contrib = [(feature_names[i], coefs[i]) for i in indices]

        token_contrib = sorted(token_contrib, key=lambda x: abs(x[1]), reverse=True)[:10]
        if not token_contrib:
            st.info("No important tokens found for this URL.")
            return

        df = pd.DataFrame(token_contrib, columns=["Token", "Weight"])

        fig, ax = plt.subplots()
        color = "green" if label == "good" else "red"
        df.set_index("Token").plot(
            kind="barh", ax=ax, legend=False, color=color
        )
        ax.set_xlabel("Contribution Weight")
        st.pyplot(fig)
    else:
        st.info("Explainability not available for this model.")


def update_stats(url, label):
    """Update counts and history"""
    st.session_state.total += 1
    if label == "good":
        st.session_state.safe += 1
    else:
        st.session_state.malicious += 1
    st.session_state.history.append({"URL": url, "Prediction": label})






def main():
    st.title("🔐 Phishing Domain and Malicious URL Detection")

    menu = st.sidebar.selectbox("Choose View", ["Check URL", "Dashboard", "AI Chatbot Assistant"])

    if menu == "Check URL":
        url = st.text_input("Enter URL", "https://www.example.com/")

        if st.button("Detect!"):
            result = predictor(url)
            label = "good" if result == 0 else "bad"

            if label == "good":
                st.subheader("✅ Safe URL")
                load_lottieurl("https://lottie.host/e4735726-9f74-4286-9ee9-c393e912cc77/47OtrlTYLi.json")
            else:
                st.subheader("❌ Malicious URL")
                load_lottieurl("https://lottie.host/8ae43c45-0511-4beb-84de-e3e075a7ef0d/Xu8r4ZE9no.json")

            # Confidence score (if available)
            vectorized = cvectorizer.transform([url])
            if hasattr(model, "predict_proba"):
                conf = np.max(model.predict_proba(vectorized))
                st.progress(int(conf * 100))
                st.write(f"Confidence: {conf:.2f}")

            # Update stats
            update_stats(url, label)

            # Show explanation
            explain_prediction(url, label)

    elif menu == "Dashboard":
        st.header("📊 Dashboard")
        st.metric("Total URLs Checked", st.session_state.total)
        st.metric("Safe URLs", st.session_state.safe)
        st.metric("Malicious URLs", st.session_state.malicious)

        # Bar chart
        data = {"Safe": st.session_state.safe, "Malicious": st.session_state.malicious}
        st.bar_chart(pd.DataFrame.from_dict(data, orient="index", columns=["Count"]))

    elif menu == "AI Chatbot Assistant":
        st.header("💬 AI Chatbot Assistant ")

        # Initialize chat history if not present
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # User input
        user_q = st.text_input("What would you like to know?", key="chat_input")

        if st.button("Send", key="send_chat"):
            if user_q:
                reply = chatbot_response(user_q)  # Call your AI function
                st.session_state.chat_history.append(("You", user_q))
                st.session_state.chat_history.append(("AI", reply))
                st.rerun()  # Refresh to show new messages

    # Show conversation in the main page
        chat_display = "\n".join([f"{speaker}: {msg}" for speaker, msg in st.session_state.chat_history])
        st.text_area("Conversation", value=chat_display, height=300)

    # Optional: reset button
        if st.button("Reset Conversation", key="reset_chat"):
            st.session_state.chat_history = []
            st.rerun()


if __name__ == "__main__":
    main()