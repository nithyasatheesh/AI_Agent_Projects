# app.py

import streamlit as st
from openai import OpenAI

# OpenAI client
client = OpenAI()

# Allowed technical keywords
ALLOWED_TOPICS = [
    "python",
    "java",
    "spring boot",
    "api",
    "json",
    "unit test",
    "testing",
    "debug",
    "bug",
    "compile",
    "exception",
    "fastapi",
    "flask",
    "sql",
    "database",
    "git",
    "github",
    "docker",
    "kubernetes",
    "ci/cd",
    "jenkins",
    "microservices",
    "rest",
    "backend",
    "frontend",
    "javascript",
    "react",
    "code",
    "programming",
    "algorithm",
    "data structure"
]


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

        Restrictions:
        - ONLY answer software engineering and technical questions
        - Reject generic/non-technical questions
        - Do not answer weather, politics, sports, movies,
          personal advice, or casual chat
        """

    def is_technical_question(self, user_query: str) -> bool:

        query = user_query.lower()

        for keyword in ALLOWED_TOPICS:
            if keyword in query:
                return True

        return False

    def get_response(self, user_query: str):

        # Reject non-technical questions
        if not self.is_technical_question(user_query):

            return (
                "I am an AI Coding Tutor specialized only in "
                "software engineering and programming-related questions.\n\n"
                "Please ask a coding, debugging, testing, or technical question."
            )

        # OpenAI response
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": user_query
                }
            ],
            temperature=0.3
        )

        return response.choices[0].message.content


# ---------------- STREAMLIT UI ---------------- #

st.set_page_config(
    page_title="AI Coding Tutor",
    page_icon="💻",
    layout="centered"
)

st.title("💻 AI Coding Tutor")

st.markdown("""
Enterprise onboarding and coding assistant.

### Supported Areas
- Programming concepts
- Debugging support
- Unit testing
- APIs & backend development
- Spring Boot / Python / SQL
- CI/CD & DevOps basics

### Restrictions
- No weather/news/general questions
- No casual conversation
""")

# Initialize tutor
tutor = AICodingTutor()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display old messages
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Ask a coding or technical question...")

if user_input:

    # Store user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    answer = tutor.get_response(user_input)

    # Store assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    # Show assistant response
    with st.chat_message("assistant"):
        st.markdown(answer)
