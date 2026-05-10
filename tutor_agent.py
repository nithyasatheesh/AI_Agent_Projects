# app.py

import tempfile

import matplotlib.pyplot as plt
import streamlit as st
from gtts import gTTS
from openai import OpenAI

# ---------------- OPENAI CLIENT ---------------- #

client = OpenAI()

# ---------------- CONFIG ---------------- #

ALLOWED_TOPICS = [

    # Programming
    "python",
    "java",
    "javascript",
    "react",
    "spring boot",
    "fastapi",
    "flask",
    "api",
    "backend",
    "frontend",
    "programming",
    "code",

    # Testing & Debugging
    "unit test",
    "testing",
    "debug",
    "bug",
    "exception",
    "compile",

    # Databases
    "sql",
    "database",
    "mysql",
    "postgresql",

    # DevOps
    "docker",
    "kubernetes",
    "ci/cd",
    "jenkins",
    "git",
    "github",

    # CS Concepts
    "algorithm",
    "data structure",
    "oop",
    "object oriented",

    # Data Science / AI
    "machine learning",
    "deep learning",
    "data science",
    "data analysis",
    "pandas",
    "numpy",
    "outlier",
    "outliers",
    "missing value",
    "missing values",
    "normalization",
    "standardization",
    "feature engineering",
    "classification",
    "regression",
    "clustering",
    "dataset",
    "statistics",
    "iqr",
    "quartile",

    # Cloud
    "aws",
    "azure",
    "gcp"
]

FOLLOW_UP_WORDS = [
    "yes",
    "no",
    "continue",
    "show example",
    "example",
    "proceed",
    "explain more",
    "show in python",
    "show in java",
    "show code"
]

# ---------------- AI TUTOR ---------------- #

class AICodingTutor:

    def __init__(self):

        self.system_prompt = """
        You are an AI Coding Tutor designed for enterprise onboarding
        and software engineering bootcamps.

        Responsibilities:
        - Help users solve coding problems
        - Explain technical concepts
        - Debug errors
        - Guide users step-by-step
        - Encourage learning instead of directly giving full solutions

        Behavior Rules:
        - Be concise, professional, and technical
        - Encourage users to learn through explanations
        - Ask guided follow-up questions when appropriate
        - Maintain conversational context

        Formatting Rules:
        - Use clean markdown formatting
        - Avoid raw LaTeX or mathematical markup
        - Prefer readable plain-text formulas
        - Use bullet points and numbered steps
        - Keep explanations visually clean

        STRICT RESTRICTIONS:
        - ONLY answer software engineering, programming,
          debugging, testing, DevOps, cloud, data science,
          AI/ML, and technical questions
        - Reject generic/non-technical questions

        DO NOT answer:
        - Weather
        - Politics
        - Sports
        - Movies
        - Health advice
        - Casual conversation
        - General non-technical topics
        """

    def is_technical_question(self, user_query: str):

        query = user_query.lower().strip()

        # Allow follow-up replies
        if query in FOLLOW_UP_WORDS:
            return True

        # Check keywords
        for keyword in ALLOWED_TOPICS:
            if keyword in query:
                return True

        return False

    def get_response(self, messages):

        latest_user_message = messages[-1]["content"]

        # Reject non-technical questions
        if not self.is_technical_question(latest_user_message):

            return (
                "I am an AI Coding Tutor specialized only in "
                "software engineering and technical topics.\n\n"
                "Please ask a coding, debugging, testing, "
                "data science, cloud, or programming-related question."
            )

        # Build history
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

        # Generate response
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=chat_messages,
            temperature=0.3
        )

        return response.choices[0].message.content


# ---------------- AUDIO SUMMARY ---------------- #

def generate_audio_summary(text):

    summary_prompt = f"""
    Summarize this explanation into
    3 concise learning points.

    Keep it short and easy to understand.

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

        tts = gTTS(text=summary, lang="en")

        temp_audio = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp3"
        )

        tts.save(temp_audio.name)

        return temp_audio.name, summary

    except Exception as e:

        return None, f"Audio generation failed: {str(e)}"


# ---------------- VISUALIZATION ---------------- #

def show_outlier_visualization():

    data = [10, 12, 13, 14, 15, 16, 18, 20, 100]

    fig, ax = plt.subplots()

    ax.boxplot(data)

    ax.set_title("Outlier Detection Example")

    st.pyplot(fig)


# ---------------- STREAMLIT UI ---------------- #

st.set_page_config(
    page_title="AI Coding Tutor",
    page_icon="💻",
    layout="wide"
)

st.title("💻 AI Coding Tutor")

st.markdown("""
Enterprise onboarding and coding assistant.

### Features
- Technical Q&A
- Guided learning
- Audio learning summaries
- Visualization support
- File upload support
- Multi-line coding questions
""")

# Initialize tutor
tutor = AICodingTutor()

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- FILE UPLOAD ---------------- #

uploaded_file = st.file_uploader(
    "📂 Upload a technical document or code file",
    type=["txt", "py", "java", "md"]
)

file_content = ""

if uploaded_file is not None:

    file_content = uploaded_file.read().decode("utf-8")

    st.markdown("### 📄 Uploaded File Preview")

    st.code(file_content[:2000])

# ---------------- CHAT HISTORY ---------------- #

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- MULTI-LINE INPUT ---------------- #

st.markdown("### 💬 Ask Your Question")

user_input = st.text_area(
    "Enter your coding or technical question:",
    height=150,
    placeholder="""
Example:
- Explain outlier detection using Python
- Help debug this Spring Boot error
- Explain CI/CD pipeline
"""
)

# Submit button
submit = st.button("🚀 Ask Tutor")

# ---------------- PROCESS REQUEST ---------------- #

if submit and user_input:

    final_input = user_input

    # Append uploaded file content if present
    if file_content:

        final_input += f"""

Uploaded File Content:
{file_content}
"""

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": final_input
    })

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate answer
    answer = tutor.get_response(
        st.session_state.messages
    )

    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    # Display assistant response
    with st.chat_message("assistant"):

        st.markdown(answer)

        # Visualization
        if (
            "outlier" in user_input.lower()
            or "iqr" in user_input.lower()
        ):

            st.markdown("### 📊 Visualization")

            show_outlier_visualization()

        # Audio Summary
        st.markdown("### 🔊 Audio Learning Summary")

        audio_file, summary = generate_audio_summary(answer)

        st.markdown(summary)

        if audio_file:
            audio_bytes = open(audio_file, "rb").read()
            st.audio(audio_bytes, format="audio/mp3")
