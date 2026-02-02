import streamlit as st
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_classic.chains import LLMMathChain, LLMChain
from langchain_classic.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
import numexpr as ne
# -------------------------------------------------
# Streamlit Config
# -------------------------------------------------
st.set_page_config(
    page_title="Text To Math Problem Solver",
    page_icon="🔢"
)
st.title("🔢 Text To Math & Reasoning Assistant (llama-3.1-8b-instant + Groq)")

# -------------------------------------------------
# Groq API Key Input
# -------------------------------------------------
groq_api_key = st.sidebar.text_input(
    "Groq API Key",
    type="password"
)
if not groq_api_key:
    st.info("Please add your Groq API key to continue.")
    st.stop()

# -------------------------------------------------
# LLM Initialization
# -------------------------------------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=groq_api_key
)

# -------------------------------------------------
# TOOLS
# -------------------------------------------------

# 1️⃣ Wikipedia Search Tool
wiki = WikipediaAPIWrapper()

@tool(description="Search Wikipedia for factual information")
def wikipedia_search(query: str) -> str:
    """Search Wikipedia and return a summary of the query."""
    return wiki.run(query)

# 2️⃣ Calculator Tool
@tool(description="Evaluate numeric expressions safely using Python's numexpr library")
def calculator(expression: str) -> str:
    """Evaluate numeric math expressions and return the result as a string."""
    try:
        return str(ne.evaluate(expression))
    except Exception:
        return "Invalid expression"



# 3️⃣ Reasoning Tool
reasoning_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
You are a math assistant. For calculations, only generate numeric expressions suitable for the calculator tool.

Question: {question}
Answer:
"""
)
reasoning_chain = LLMChain(llm=llm, prompt=reasoning_prompt)

@tool(description="Solve logic and reasoning questions with explanation")
def reasoning_tool(question: str) -> str:
    """Solve logic and reasoning questions step by step."""
    return reasoning_chain.run(question)

# -------------------------------------------------
# AGENT
# -------------------------------------------------
agent = create_agent(
    model=llm,
    tools=[wikipedia_search, calculator, reasoning_tool]
)

# -------------------------------------------------
# Chat History
# -------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi 👋 I'm a math & reasoning assistant. Ask me anything!"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# -------------------------------------------------
# User Input
# -------------------------------------------------
question = st.text_area(
    "Enter your question:",
    "I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. "
    "Then I buy a dozen apples and 2 packs of blueberries. Each pack has 25 berries. "
    "How many total fruits do I have?"
)

# -------------------------------------------------
# Run Agent and Display Clean Answer
# -------------------------------------------------
if st.button("Find my answer"):
    if question.strip():
        st.session_state.messages.append({"role": "user", "content": question})
        st.chat_message("user").write(question)

        with st.spinner("Thinking..."):
            # Invoke agent correctly inside spinner
            response = agent.invoke({
                "messages": [{"role": "user", "content": question}]
            })
            # Extract only the final AI message
            final_answer = None
            for msg in reversed(response["messages"]):
    # AIMessage or ToolMessage from LangChain
                from langchain_classic.schema import AIMessage

                if isinstance(msg, AIMessage):
                    if msg.content.strip():  # non-empty content
                        final_answer = msg.content
                break



            if final_answer:
                st.session_state.messages.append({"role": "assistant", "content": final_answer})
                st.chat_message("assistant").write(final_answer)
            else:
                st.warning("No answer could be extracted from the model response.")
    else:
        st.warning("Please enter a question.")


