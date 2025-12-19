import streamlit as st
import os
import sys
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.agent import MeteorologyAgent
from src.utils.data_loader import load_data
from src.config import logger

# Page Config
st.set_page_config(
    page_title="Meteorology Analyzer",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "df" not in st.session_state:
    st.session_state.df = None
if "api_key" not in st.session_state:
    st.session_state.api_key = None

# Helper Functions
def clear_chat():
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.title("🌤️ MetAnalyzer")
    st.markdown("---")
    
    # API Key Input
    api_key_input = st.text_input(
        "🔑 DeepSeek API Key", 
        type="password", 
        help="Enter your API Key here. It will not be stored permanently.",
        value=st.session_state.api_key if st.session_state.api_key else ""
    )
    if api_key_input:
        st.session_state.api_key = api_key_input
        # Set env var temporarily for this session
        os.environ["DEEPSEEK_API_KEY"] = api_key_input
    
    st.markdown("---")

    # File Uploader
    uploaded_file = st.file_uploader("Upload Station Data (CSV)", type=['csv', 'txt'])
    if uploaded_file:
        try:
            with st.spinner("Loading data..."):
                # Use the new data loader
                df = load_data(uploaded_file)
                st.session_state.df = df
                st.success(f"✅ Loaded {len(df)} records")
                st.markdown(f"**Columns:** {', '.join(df.columns[:5])}...")
        except Exception as e:
            st.error(f"Failed to load data: {e}")

    st.markdown("---")
    st.markdown("### 🛠️ Actions")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Stats", use_container_width=True):
             if st.session_state.df is not None:
                st.session_state.messages.append({"role": "user", "content": "Show me the dataset statistics."})
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": "Here are the statistics:", 
                    "type": "dataframe",
                    "dataframe": st.session_state.df.describe()
                })
             else:
                st.warning("Load data first!")
    
    with col2:
        st.button("🗑️ Clear", on_click=clear_chat, use_container_width=True)

# Initialize Agent (Re-initialize if API Key changes or first run)
# Note: st.cache_resource is good, but we need to invalidate it if API key changes.
# For simplicity, let's create a wrapper that checks the key.
@st.cache_resource(show_spinner=False)
def get_agent_instance(api_key_signature):
    # api_key_signature is just to force recompilation when key changes
    return MeteorologyAgent()

# Pass key signature to force reload if key changes
agent = get_agent_instance(st.session_state.api_key)


# Main Chat Interface
st.title("💬 Meteorology Assistant")

# Welcome Message
if st.session_state.df is None:
    st.info("👋 Welcome! Please upload a weather station CSV file in the sidebar to get started.")
    if not st.session_state.api_key:
        st.warning("⚠️ Don't forget to enter your DeepSeek API Key in the sidebar!")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Display Thought if available (for assistant)
        if message.get("thought"):
            with st.expander("💭 Thought Process", expanded=False):
                st.markdown(f"**Reasoning:** {message['thought']}")
                if message.get("action") and message.get("action") != "None":
                    st.markdown(f"**Action:** `{message['action']}`")
        
        # Display Content
        if message.get("type") == "dataframe" and "dataframe" in message:
            st.markdown(message["content"])
            st.dataframe(message["dataframe"])
        else:
            st.markdown(message["content"])
        
        # Display Figure if available
        if message.get("figure"):
            st.pyplot(message["figure"])

# Chat Input
if prompt := st.chat_input("Ask about weather data (e.g., 'Plot temperature for Lanzhou')..."):
    if st.session_state.df is None:
        st.error("⚠️ Please upload a data file first.")
    elif not st.session_state.api_key:
        st.error("⚠️ Please enter your API Key in the sidebar.")
    else:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            try:
                # Use st.status for better UX
                with st.status("🧠 Agent is thinking...", expanded=True) as status:
                    st.write("Analyzing request...")
                    
                    # Call the stateless agent
                    response = agent.run(
                        query=prompt, 
                        df=st.session_state.df, 
                        history=st.session_state.messages
                    )
                    
                    thought = response.get("thought", "")
                    action = response.get("action", "")
                    result = response.get("result", "")
                    figure = response.get("figure", None)
                    
                    if action and action != "None":
                        st.write(f"Executing tool: `{action}`")
                    
                    status.update(label="✅ Analysis Complete", state="complete", expanded=False)

                # Display Thought (Collapsed by default after status closes)
                if thought:
                    with st.expander("💭 Thought Process", expanded=False):
                        st.markdown(f"**Reasoning:** {thought}")
                        st.markdown(f"**Action:** `{action}`")

                # Display Result
                st.markdown(result)
                
                # Display Figure
                if figure:
                    st.pyplot(figure)
                
                # Save to history
                msg_data = {
                    "role": "assistant",
                    "content": result,
                    "thought": thought,
                    "action": action,
                    "figure": figure
                }
                st.session_state.messages.append(msg_data)
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
                logger.exception("App Error")
