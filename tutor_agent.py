# app.py

import tempfile
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from gtts import gTTS
from openai import OpenAI

# ---------------- OPENAI CLIENT ---------------- #

client = OpenAI()

# ---------------- CONFIG ---------------- #

ALLOWED_TOPICS = [

    # Programming
    "python", "java", "javascript", "react",
    "spring boot", "fastapi", "flask",
    "api", "backend", "frontend",
    "programming", "code",

    # Debugging & Testing
    "unit test", "testing", "debug",
    "bug", "exception", "compile",

    # Database
    "sql", "database", "mysql", "postgresql",

    # DevOps
    "docker", "kubernetes", "ci/cd",
    "jenkins", "git", "github",

    # Data Science
    "data science", "machine learning",
    "outlier", "outliers",
    "missing value", "missing values",
    "dataset", "visualize", "plot",
    "graph", "chart", "correlation",
    "distribution", "statistics",
    "pandas", "numpy", "iqr"
]

FOLLOW_UP_WORDS = [
    "yes",
    "no",
    "continue",
    "show example",
    "show in python",
    "show code",
    "visualize it"
]

# ---------------- AI TUTOR ---------------- #

class AICodingTutor:

    def __init__(self):

        self.system_prompt = """
        You are an AI Technical Learning Assistant.

        Responsibilities:
        - Explain coding concepts
        - Help debug errors
        - Guide users step-by-step
        - Explain datasets and visualizations
        - Help with software engineering and data science

        Formatting Rules:
        - Use clean markdown
        - Avoid raw LaTeX
        - Use numbered steps
        - Keep responses readable

        STRICT RESTRICTIONS:
        - ONLY answer technical questions
        - Reject weather/news/general chat
        """

    def is_technical_question(self, query):

        query = query.lower().strip()

        if query in FOLLOW_UP_WORDS:
            return True

        for keyword in ALLOWED_TOPICS:
            if keyword in query:
                return True

        return False

    def get_response(self, messages):

        latest_message = messages[-1]["content"]

        if not self.is_technical_question(latest_message):

            return (
                "I am an AI Technical Learning Assistant "
                "specialized only in coding, software engineering, "
                "data science, and technical topics."
            )

        chat_messages = [
            {
                "role": "system",
                "content": self.system_prompt
            }
        ]

        for msg in messages:
            chat_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=chat_messages,
            temperature=0.3
        )

        return response.choices[0].message.content


# ---------------- AUDIO SUMMARY ---------------- #

def generate_audio_summary(text):

    summary_prompt = f"""
    Summarize this into 3 concise learning points.

    Text:
    {text}
    """

    summary_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": summary_prompt
            }
        ]
    )

    summary = summary_response.choices[0].message.content

    try:

        tts = gTTS(summary)

        temp_audio = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp3"
        )

        tts.save(temp_audio.name)

        return temp_audio.name, summary

    except Exception as e:

        return None, str(e)


# ---------------- VISUALIZATION ---------------- #

def visualize_dataset(df, user_query):

    query = user_query.lower()

    st.markdown("### 📊 Dataset Visualization")

    # Missing values
    if "missing" in query:

        missing = df.isnull().sum()

        fig, ax = plt.subplots()

        missing.plot(
            kind="bar",
            ax=ax
        )

        ax.set_title("Missing Values")

        st.pyplot(fig)

    # Correlation
    elif "correlation" in query:

        corr = df.corr(numeric_only=True)

        st.dataframe(corr)

    # Distribution
    elif "distribution" in query:

        numeric_cols = df.select_dtypes(
            include="number"
        ).columns

        if len(numeric_cols) > 0:

            col = numeric_cols[0]

            fig, ax = plt.subplots()

            df[col].plot(
                kind="hist",
                bins=20,
                ax=ax
            )

            ax.set_title(f"Distribution of {col}")

            st.pyplot(fig)

    # Outlier visualization
    elif "outlier" in query or "iqr" in query:

        numeric_cols = df.select_dtypes(
            include="number"
        ).columns

        if len(numeric_cols) > 0:

            col = numeric_cols[0]

            fig, ax = plt.subplots()

            ax.boxplot(df[col].dropna())

            ax.set_title(f"Outlier Detection - {col}")

            st.pyplot(fig)

    # Generic dataframe preview
    else:

        st.dataframe(df.head())


# ---------------- STREAMLIT UI ---------------- #

st.set_page_config(
    page_title="AI Technical Learning Assistant",
    page_icon="💻",
    layout="wide"
)

st.title("💻 AI Technical Learning Assistant")

st.markdown("""
### Features
- Coding help
- Debugging support
- Dataset analysis
- Visualization generation
- Audio learning summaries
- File upload support
""")

# ---------------- INIT ---------------- #

tutor = AICodingTutor()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "df" not in st.session_state:
    st.session_state.df = None

# ---------------- FILE UPLOAD ---------------- #

uploaded_file = st.file_uploader(
    "📂 Upload Dataset or Code File",
    type=["csv", "txt", "py", "java"]
)

file_content = ""

if uploaded_file is not None:

    file_name = uploaded_file.name

    # CSV dataset
    if file_name.endswith(".csv"):

        df = pd.read_csv(uploaded_file)

        st.session_state.df = df

        st.success("Dataset uploaded successfully!")

        st.markdown("### 📄 Dataset Preview")

        st.dataframe(df.head())

    # Text/code file
    else:

        file_content = uploaded_file.read().decode("utf-8")

        st.markdown("### 📄 File Preview")

        st.code(file_content[:2000])

# ---------------- CHAT HISTORY ---------------- #

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- PROCESS REQUEST ---------------- #

if "pending_input" in st.session_state:

    user_input = st.session_state.pending_input

    final_input = user_input

    # Attach code/text file content
    if file_content:

        final_input += f"""

Uploaded File Content:
{file_content}
"""

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Generate AI response
    answer = tutor.get_response(
        st.session_state.messages
    )

    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    # Clear pending state
    del st.session_state.pending_input

    st.rerun()

# ---------------- SHOW LATEST FEATURES ---------------- #

if st.session_state.messages:

    latest_msg = st.session_state.messages[-1]

    if latest_msg["role"] == "assistant":

        latest_answer = latest_msg["content"]

        # DATASET VISUALIZATION
        if st.session_state.df is not None:

            visualize_dataset(
                st.session_state.df,
                latest_answer
            )

        # AUDIO SUMMARY
        st.markdown("### 🔊 Audio Learning Summary")

        audio_file, summary = generate_audio_summary(
            latest_answer
        )

        st.markdown(summary)

        if audio_file:

            audio_bytes = open(audio_file, "rb").read()

            st.audio(
                audio_bytes,
                format="audio/mp3"
            )

# ---------------- INPUT FORM ---------------- #

with st.form("chat_form", clear_on_submit=True):

    st.markdown("### 💬 Ask Your Question")

    user_input = st.text_area(
        "Enter your coding or dataset question:",
        height=150,
        placeholder="""
Examples:
- Visualize missing values
- Detect outliers
- Explain CI/CD pipeline
- Debug this Python code
"""
    )

    submit = st.form_submit_button("🚀 Ask Assistant")

    if submit and user_input:

        st.session_state.pending_input = user_input

        st.rerun()
