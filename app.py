import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
from openai import OpenAI

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Bolo - AI Doubt Solver", layout="wide")

# ---------- SETUP: AI CLIENT ----------
# Groq gives free, very fast access to open models, using an OpenAI-compatible API.
# Get a free key at https://console.groq.com -> API Keys
client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY", ""),
    base_url="https://api.groq.com/openai/v1",
)

DATA_FILE = "doubts_log.csv"


# ---------- DATA HANDLING ----------
def load_log():
    """Load past doubts from the CSV file, or start a fresh empty table."""
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    return pd.DataFrame(columns=["timestamp", "question", "language", "subject", "topic"])


def save_entry(question, language, subject, topic):
    """Add one new doubt to the log and save it back to disk."""
    df = load_log()
    new_row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "question": question,
        "language": language,
        "subject": subject,
        "topic": topic,
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)


# ---------- AI CALL ----------
def get_explanation(question, language):
    """Ask the AI to explain the doubt, and to tag its subject/topic for analytics."""
    prompt = f"""You are a friendly teacher explaining concepts to a student.
Explain the following doubt in {language}, using simple words and one relatable analogy.
Keep it under 150 words.

Also return a JSON tag for the subject and specific topic.

Doubt: {question}

Respond ONLY with valid JSON in this exact format, no extra text before or after:
{{"explanation": "...", "subject": "...", "topic": "..."}}
"""
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback if the model doesn't return clean JSON - app still works.
        data = {"explanation": raw, "subject": "Unknown", "topic": "Unknown"}
    return data


# ---------- UI: HEADER ----------
st.title("📚 Bolo — AI Doubt Solver in Your Language")
st.caption("Ask any doubt. Get it explained the way a teacher would, in your language.")

tab1, tab2 = st.tabs(["Ask a Doubt", "📊 Dashboard"])

# ---------- TAB 1: ASK A DOUBT ----------
with tab1:
    st.subheader("Ask your doubt")
    question = st.text_area(
        "Type your question here",
        placeholder="e.g. Why does ice float on water?",
    )
    language = st.selectbox("Explain in:", ["Hindi", "Hinglish", "English"])

    if st.button("Get Explanation"):
        if question.strip() == "":
            st.warning("Please type a question first.")
        else:
            with st.spinner("Thinking..."):
                result = get_explanation(question, language)
            st.success(result["explanation"])
            st.caption(f"Subject: {result['subject']} | Topic: {result['topic']}")
            save_entry(question, language, result["subject"], result["topic"])

# ---------- TAB 2: DASHBOARD ----------
with tab2:
    st.subheader("Doubt Analytics")
    df = load_log()

    if df.empty:
        st.info("No doubts logged yet. Ask a question in the first tab!")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.write("Most asked subjects")
            st.bar_chart(df["subject"].value_counts())
        with col2:
            st.write("Language usage")
            st.bar_chart(df["language"].value_counts())

        st.write("Recent doubts")
        st.dataframe(df.tail(10))
