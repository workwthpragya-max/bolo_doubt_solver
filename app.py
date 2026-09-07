import streamlit as st
import pandas as pd
import json
import os
import io
import uuid
from datetime import datetime
from openai import OpenAI
from gtts import gTTS

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

    /* Dashboard cards - with 3D depth + hover tilt illusion */
    div[data-testid="column"] {
        background: linear-gradient(145deg, #1a1613, #100d0b);
        padding: 22px;
        border-radius: 18px;
        border: 1px solid #241f1b;
        box-shadow:
            0 8px 24px rgba(0,0,0,0.5),
            inset 0 1px 0 rgba(255,255,255,0.03);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    div[data-testid="column"]:hover {
        transform: perspective(800px) rotateX(2deg) translateY(-4px);
        box-shadow:
            0 16px 32px rgba(255,106,43,0.15),
            0 8px 24px rgba(0,0,0,0.6),
            inset 0 1px 0 rgba(255,255,255,0.05);
    }

    /* Glow behind the hero headline for depth */
    h1 {
        text-shadow: 0 0 40px rgba(255,106,43,0.15);
    }

    /* Stat numbers get a subtle glow + lift on hover */
    .stat-number {
        transition: transform 0.2s ease, text-shadow 0.2s ease;
        display: inline-block;
    }

    /* Text area gets an inset "pressed into surface" 3D feel */
    .stTextArea textarea {
        box-shadow: inset 0 2px 8px rgba(0,0,0,0.4) !important;
    }

    /* Buttons get raised 3D depth */
    .stButton > button {
        box-shadow: 0 4px 0 #b8471a, 0 6px 16px rgba(255,106,43,0.25);
    }
    .stButton > button:active {
        transform: translateY(3px);
        box-shadow: 0 1px 0 #b8471a, 0 2px 8px rgba(255,106,43,0.25);
    }

    /* Sidebar buttons - force black text so it's readable on the orange background */
    section[data-testid="stSidebar"] .stButton > button {
        color: #000000 !important;
        font-weight: 700 !important;
    }

    .stDataFrame {
        border-radius: 14px;
        overflow: hidden;
        box-shadow: 0 8px 20px rgba(0,0,0,0.4);
    }
</style>
""", unsafe_allow_html=True)

# ---------- SETUP: AI CLIENT ----------
client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY", ""),
    base_url="https://api.groq.com/openai/v1",
)

DATA_FILE = "doubts_log.csv"
CHATS_FILE = "chats.json"


# ---------- CHAT PERSISTENCE (saves conversations to disk) ----------
def load_chats_from_disk():
    if os.path.exists(CHATS_FILE):
        with open(CHATS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_chats_to_disk():
    with open(CHATS_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state.chats, f, ensure_ascii=False, indent=2)


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


# ---------- AI CALL (now conversation-aware) ----------
def get_explanation(question, language, conversation_context=""):
    prompt = f"""You are a friendly teacher having an ongoing conversation with a student.
Explain things in {language}, using simple words and relatable analogies.
Keep each answer under 150 words.

If there is previous conversation, use it to understand follow-up questions
(e.g. "why?" or "give an example" refers back to what was just discussed).

Previous conversation so far:
{conversation_context if conversation_context else "(this is the first question)"}

New question from the student: {question}

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


# ---------- VOICE: SPEECH-TO-TEXT ----------
def transcribe_audio(audio_bytes):
    """Send recorded audio to Groq's Whisper model and get back the transcribed text."""
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "doubt.wav"
    transcript = client.audio.transcriptions.create(
        model="whisper-large-v3",
        file=audio_file,
    )
    return transcript.text


# ---------- VOICE: TEXT-TO-SPEECH ----------
LANG_CODE_MAP = {
    "Hindi": "hi",
    "Hinglish": "hi",   # gTTS has no Hinglish voice; Hindi voice reads it naturally enough
    "Kannada": "kn",
    "Marathi": "mr",
    "English": "en",
}

def text_to_speech(text, language):
    lang_code = LANG_CODE_MAP.get(language, "en")
    tts = gTTS(text=text, lang=lang_code)
    audio_fp = io.BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    return audio_fp


# ---------- UI: HERO HEADER ----------
st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)
st.markdown("<h1>Doubts, <span class='accent'>solved</span><br>in your language.</h1>", unsafe_allow_html=True)
st.caption("Bolo explains any academic question in Hindi, Hinglish, Kannada, Marathi, or English — like a teacher would.")

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
    st.markdown("<div class='stat-number'>5</div>", unsafe_allow_html=True)
    st.markdown("<div class='stat-label'>Languages Supported</div>", unsafe_allow_html=True)
    st.markdown("<div class='stat-divider'></div>", unsafe_allow_html=True)
with stat_col3:
    st.markdown(f"<div class='stat-number'>{subjects_covered}</div>", unsafe_allow_html=True)
    st.markdown("<div class='stat-label'>Subjects Covered</div>", unsafe_allow_html=True)
    st.markdown("<div class='stat-divider'></div>", unsafe_allow_html=True)

st.write("")
st.write("")

tab1, tab2 = st.tabs(["✏️ Ask a Doubt", "📊 Dashboard"])

# ---------- MULTI-CHAT MEMORY SETUP (loaded from disk so it survives new windows) ----------
if "chats" not in st.session_state:
    st.session_state.chats = load_chats_from_disk()

if "active_chat_id" not in st.session_state:
    non_empty_chats = [cid for cid, c in st.session_state.chats.items() if c["messages"]]
    if non_empty_chats:
        st.session_state.active_chat_id = non_empty_chats[-1]  # open the most recent one
    else:
        new_id = str(uuid.uuid4())
        st.session_state.chats[new_id] = {"title": "New Chat", "messages": []}
        st.session_state.active_chat_id = new_id


def start_new_chat():
    new_id = str(uuid.uuid4())
    st.session_state.chats[new_id] = {"title": "New Chat", "messages": []}
    st.session_state.active_chat_id = new_id
    save_chats_to_disk()


def delete_chat(chat_id):
    del st.session_state.chats[chat_id]
    if st.session_state.active_chat_id == chat_id:
        remaining = list(st.session_state.chats.keys())
        if remaining:
            st.session_state.active_chat_id = remaining[-1]
        else:
            start_new_chat()
    save_chats_to_disk()


# ---------- SIDEBAR: PAST CHATS ----------
with st.sidebar:
    st.markdown("### 💬 Your Conversations")
    if st.button("➕ New Chat", use_container_width=True):
        start_new_chat()
        st.rerun()

    st.write("")
    # Show most recent chats first — skip empty conversations that never got a question
    for chat_id in reversed(list(st.session_state.chats.keys())):
        chat = st.session_state.chats[chat_id]
        if not chat["messages"]:
            continue
        is_active = chat_id == st.session_state.active_chat_id
        label = ("🟠 " if is_active else "") + chat["title"]

        col_chat, col_del = st.columns([5, 1])
        with col_chat:
            if st.button(label, key=f"chat_btn_{chat_id}", use_container_width=True):
                st.session_state.active_chat_id = chat_id
                st.rerun()
        with col_del:
            if st.button("🗑️", key=f"del_btn_{chat_id}"):
                delete_chat(chat_id)
                st.rerun()

# ---------- TAB 1: ASK A DOUBT (chat-style, supports follow-ups) ----------
with tab1:
    st.write("")

    active_chat = st.session_state.chats[st.session_state.active_chat_id]

    language = st.selectbox("Explain in:", ["Hindi", "Hinglish", "Kannada", "Marathi", "English"])

    def build_context():
        """Turn the last few messages into a text block the AI can read for follow-up context."""
        recent = active_chat["messages"][-6:]  # last 6 messages is plenty of context
        lines = []
        for m in recent:
            speaker = "Student" if m["role"] == "user" else "Teacher"
            lines.append(f"{speaker}: {m['content']}")
        return "\n".join(lines)

    def handle_new_question(question_text):
        active_chat["messages"].append({"role": "user", "content": question_text})

        # Name the chat after the first question, so it's recognizable in the sidebar
        if active_chat["title"] == "New Chat":
            short_title = question_text.strip()[:40]
            active_chat["title"] = short_title + ("..." if len(question_text.strip()) > 40 else "")

        context = build_context()
        with st.spinner("Thinking..."):
            result = get_explanation(question_text, language, context)
        active_chat["messages"].append({
            "role": "assistant",
            "content": result["explanation"],
            "subject": result["subject"],
            "topic": result["topic"],
        })
        save_entry(question_text, language, result["subject"], result["topic"])
        save_chats_to_disk()

    # Show the whole conversation so far
    for i, msg in enumerate(active_chat["messages"]):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                st.caption(f"📌 Subject: {msg.get('subject', '')} • Topic: {msg.get('topic', '')}")
                audio_key = f"audio_{st.session_state.active_chat_id}_{i}"
                if st.button("🔊 Listen", key=f"listen_{audio_key}"):
                    with st.spinner("Preparing audio..."):
                        st.session_state[audio_key] = text_to_speech(msg["content"], language)
                if audio_key in st.session_state:
                    st.audio(st.session_state[audio_key], format="audio/mp3")

    if not active_chat["messages"]:
        st.info("This is a new conversation. Ask your first doubt below!")

    # Voice input, tucked away so text stays the primary flow
    with st.expander("🎤 Or record your doubt instead of typing"):
        audio_value = st.audio_input("Record")
        if audio_value is not None:
            with st.spinner("Listening..."):
                voice_text = transcribe_audio(audio_value.read())
            st.markdown(f"**Heard:** {voice_text}")
            if st.button("Send this question"):
                handle_new_question(voice_text)
                st.rerun()

    # Chat-style text input - works for both the first question and any follow-up
    user_input = st.chat_input("Ask a doubt, or ask a follow-up question...")
    if user_input:
        handle_new_question(user_input)
        st.rerun()



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