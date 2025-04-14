import os
# os.environ["STREAMLIT_WATCHDOG_MODE"] = "none"

import streamlit as st
import logging
from create_agent import create_agent

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# App Title
st.title("Legal Help Assistant")

# Ensure agent is initialized once when the app starts
if "agent" not in st.session_state:
    logging.info("Initializing AI agent.")
    st.session_state.agent = create_agent()  # Create the agent and store it in session state

# Function to process user query
def respond_to_query(query):
    """
    Processes user input by sending it to the AI agent and retrieving a response.
    """
    agent = st.session_state.agent  # Access the agent stored in session state
    logging.info(f"Processing user query: {query}")
    try:
        reply = agent.query(query)  # Assumes agent.query is synchronous or handled synchronously
        response = str(reply)
        logging.info(f"Generated response: {response}")
        return response
    except Exception as e:
        logging.error(f"Error processing query: {e}", exc_info=True)
        return "Sorry, an error occurred while processing your request."

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input
prompt = st.chat_input("How Can I Help?")

if prompt:
    logging.info(f"User input received: {prompt}")
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Get response
        response = respond_to_query(prompt)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
