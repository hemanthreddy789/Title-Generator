import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Set your Gemini API key here
os.environ["GOOGLE_API_KEY"] = 'AIzaSyCG493FtCpsEsjZpwSAn6VJHgBeYi6e-0M'

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

# Streamlit UI
st.set_page_config(page_title="AI Title Generator", page_icon="⭐")
st.title("🌟 Creative Title Generator with Gemini")

option = st.selectbox(
    "What do you need a title for?",
    ("Story", "Project", "Research Paper")
)

text_label = {
    "Story": "Paste your story or story idea here:",
    "Project": "Describe your project here:",
    "Research Paper": "Enter your research abstract or topic:"
}[option]
user_input = st.text_area(text_label)

if st.button("Generate Title"):
    if user_input.strip():
        with st.spinner("Coming up with a perfect title..."):
            prompt_texts = {
                "Story": """
You are a creative AI assistant. Read the following story or story idea and suggest a short, captivating, original title for it. Only return the title.
Story:
{content}
""",
                "Project": """
You are an inventive AI assistant. Based on the description below, suggest a short, catchy, and descriptive name for this project. Only return the name.
Project Description:
{content}
""",
                "Research Paper": """
You are an academic AI assistant. Read the abstract or details below and propose a formal, concise, and relevant research paper title. Only return the title.
Research Details:
{content}
"""
            }
            prompt = PromptTemplate(
                input_variables=["content"],
                template=prompt_texts[option]
            )
            chain = LLMChain(llm=llm, prompt=prompt)
            result = chain.run(user_input)
            st.success("Suggested Title:")
            st.markdown(f"**{result.strip()}**")
    else:
        st.warning("Please enter your content above before generating the title.")
