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
    page_title="气象分析助手",
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
    st.title("🌤️ 气象分析仪")
    st.markdown("---")
    
    # API Key Input
    api_key_input = st.text_input(
        "🔑 DeepSeek API 密钥", 
        type="password", 
        help="在此输入您的 API 密钥。密钥仅在当前会话有效，不会永久保存。",
        value=st.session_state.api_key if st.session_state.api_key else ""
    )
    if api_key_input:
        st.session_state.api_key = api_key_input
        # Set env var temporarily for this session
        os.environ["DEEPSEEK_API_KEY"] = api_key_input
    
    st.markdown("---")

    # File Uploader
    uploaded_file = st.file_uploader("上传气象站数据 (CSV)", type=['csv', 'txt'])
    if uploaded_file:
        try:
            with st.spinner("正在加载数据..."):
                # Use the new data loader
                df = load_data(uploaded_file)
                st.session_state.df = df
                st.success(f"✅ 成功加载 {len(df)} 条记录")
                st.markdown(f"**包含列名:** {', '.join(df.columns[:5])}...")
        except Exception as e:
            st.error(f"数据加载失败: {e}")

    st.markdown("---")
    st.markdown("### 🛠️ 快捷操作")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 数据统计", use_container_width=True):
             if st.session_state.df is not None:
                st.session_state.messages.append({"role": "user", "content": "请展示数据集的统计信息。"})
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": "以下是数据统计结果：", 
                    "type": "dataframe",
                    "dataframe": st.session_state.df.describe()
                })
             else:
                st.warning("请先加载数据！")
    
    with col2:
        st.button("🗑️ 清空对话", on_click=clear_chat, use_container_width=True)

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
st.title("💬 气象分析助手")

# Welcome Message
if st.session_state.df is None:
    st.info("👋 欢迎！请在左侧上传气象站 CSV 文件以开始分析。")
    if not st.session_state.api_key:
        st.warning("⚠️ 别忘了在侧边栏输入您的 DeepSeek API 密钥！")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Display Thought if available (for assistant)
        if message.get("thought"):
            with st.expander("💭 思考过程", expanded=False):
                st.markdown(f"**推理:** {message['thought']}")
                if message.get("action") and message.get("action") != "None":
                    st.markdown(f"**执行:** `{message['action']}`")
        
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
if prompt := st.chat_input("询问气象数据（例如：'绘制兰州站的气温变化曲线'）..."):
    if st.session_state.df is None:
        st.error("⚠️ 请先上传数据文件。")
    elif not st.session_state.api_key:
        st.error("⚠️ 请在侧边栏输入 API 密钥。")
    else:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            try:
                # Use st.status for better UX
                with st.status("🧠 智能体正在思考...", expanded=True) as status:
                    st.write("正在解析请求...")
                    
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
                        st.write(f"正在调用工具: `{action}`")
                    
                    status.update(label="✅ 分析完成", state="complete", expanded=False)

                # Display Thought (Collapsed by default after status closes)
                if thought:
                    with st.expander("💭 思考过程", expanded=False):
                        st.markdown(f"**推理:** {thought}")
                        st.markdown(f"**执行:** `{action}`")

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
                st.error(f"发生错误: {e}")
                logger.exception("App Error")
