import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
from openai import OpenAI

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Bolo - AI Doubt Solver", page_icon="📚", layout="wide")

# ---------- CUSTOM STYLING (dark theme, orange accent, bold type) ----------
st.markdown("""
<style>
    /* Overall app background - deep dark brown/black */
    .stApp {
        background-color: #0d0b0a;
        color: #f5f3f0;
    }

    /* Big bold headline */
    h1 {
        font-size: 3.2rem !important;
        font-weight: 800 !important;
        letter-spacing: -1.5px;
        line-height: 1.05 !important;
        color: #f5f3f0 !important;
    }

    /* Orange word highlight helper (used via markdown) */
    .accent {
        color: #ff6a2b;
    }

    .stCaption, p {
        color: #a89f98 !important;
        font-size: 16px !important;
    }

    /* Stats row - big numbers like the reference image */
    .stat-number {
        font-size: 2.6rem;
        font-weight: 800;
        color: #f5f3f0;
        line-height: 1;
    }
    .stat-label {
        color: #ff6a2b;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 6px;
    }
    .stat-divider {
        height: 2px;
        background: linear-gradient(90deg, #ff6a2b, transparent);
        margin-top: 12px;
        width: 60px;
    }

    /* Tabs styling - pill-like on dark background */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1a1613;
        padding: 6px;
        border-radius: 999px;
        width: fit-content;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 999px;
        padding: 10px 24px;
        background-color: transparent;
        font-weight: 600;
        color: #a89f98;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ff6a2b !important;
        color: #0d0b0a !important;
    }

    /* Text area and select box - dark cards */
    .stTextArea textarea {
        background-color: #1a1613 !important;
        border-radius: 16px !important;
        border: 1px solid #2e2823 !important;
        color: #f5f3f0 !important;
    }
    .stSelectbox > div > div {
        background-color: #1a1613 !important;
        border-radius: 16px !important;
        border: 1px solid #2e2823 !important;
        color: #f5f3f0 !important;
    }

    /* Buttons - bold orange, pill-shaped */
    .stButton > button {
        border-radius: 999px;
        background-color: #ff6a2b;
        color: #0d0b0a;
        font-weight: 700;
        padding: 12px 32px;
        border: none;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #ff8552;
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(255,106,43,0.35);
    }

    /* Explanation / success box */
    .stAlert {
        border-radius: 16px !important;
        border: none !important;
        background-color: #1a1613 !important;
    }

    /* Dashboard cards */
    div[data-testid="column"] {
        background-color: #151210;
        padding: 22px;
        border-radius: 18px;
        border: 1px solid #241f1b;
    }

    .stDataFrame {
        border-radius: 14px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ---------- SETUP: AI CLIENT ----------
client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY", ""),
    base_url="https://api.groq.com/openai/v1",
)

DATA_FILE = "doubts_log.csv"


# ---------- DATA HANDLING ----------
def load_log():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    return pd.DataFrame(columns=["timestamp", "question", "language", "subject", "topic"])


def save_entry(question, language, subject, topic):
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
        data = {"explanation": raw, "subject": "Unknown", "topic": "Unknown"}
    return data


# ---------- UI: HERO HEADER ----------
st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)
st.markdown("<h1>Doubts, <span class='accent'>solved</span><br>in your language.</h1>", unsafe_allow_html=True)
st.caption("Bolo explains any academic question in Hindi, Hinglish, or English — like a teacher would.")

st.write("")

# ---------- STATS ROW (dynamic, based on real logged data) ----------
df_preview = load_log()
total_doubts = len(df_preview)
subjects_covered = df_preview["subject"].nunique() if not df_preview.empty else 0

stat_col1, stat_col2, stat_col3 = st.columns(3)
with stat_col1:
    st.markdown(f"<div class='stat-number'>{total_doubts}+</div>", unsafe_allow_html=True)
    st.markdown("<div class='stat-label'>Doubts Solved</div>", unsafe_allow_html=True)
    st.markdown("<div class='stat-divider'></div>", unsafe_allow_html=True)
with stat_col2:
    st.markdown("<div class='stat-number'>3</div>", unsafe_allow_html=True)
    st.markdown("<div class='stat-label'>Languages Supported</div>", unsafe_allow_html=True)
    st.markdown("<div class='stat-divider'></div>", unsafe_allow_html=True)
with stat_col3:
    st.markdown(f"<div class='stat-number'>{subjects_covered}</div>", unsafe_allow_html=True)
    st.markdown("<div class='stat-label'>Subjects Covered</div>", unsafe_allow_html=True)
    st.markdown("<div class='stat-divider'></div>", unsafe_allow_html=True)

st.write("")
st.write("")

tab1, tab2 = st.tabs(["✏️ Ask a Doubt", "📊 Dashboard"])

# ---------- TAB 1: ASK A DOUBT ----------
with tab1:
    st.write("")
    col_left, col_right = st.columns([2, 1])

    with col_left:
        question = st.text_area(
            "Your question",
            placeholder="e.g. Why does ice float on water?",
            height=120,
        )
    with col_right:
        language = st.selectbox("Explain in:", ["Hindi", "Hinglish", "English"])
        st.write("")
        ask_clicked = st.button("Get Explanation", use_container_width=True)

    if ask_clicked:
        if question.strip() == "":
            st.warning("Please type a question first.")
        else:
            with st.spinner("Thinking..."):
                result = get_explanation(question, language)
            st.success(result["explanation"])
            st.caption(f"📌 Subject: {result['subject']} • Topic: {result['topic']}")
            save_entry(question, language, result["subject"], result["topic"])

# ---------- TAB 2: DASHBOARD ----------
with tab2:
    st.write("")
    df = load_log()

    if df.empty:
        st.info("No doubts logged yet. Ask a question in the first tab!")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Most asked subjects**")
            st.bar_chart(df["subject"].value_counts())
        with col2:
            st.markdown("**Language usage**")
            st.bar_chart(df["language"].value_counts())

        st.write("")
        st.markdown("**Recent doubts**")
        st.dataframe(df.tail(10), use_container_width=True)