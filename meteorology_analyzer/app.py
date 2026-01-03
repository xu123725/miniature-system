import streamlit as st
import os
import sys
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from src.utils.data_manager import DataManager

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
        # Set env var temporarily for this session (optional now with dependency injection)
        os.environ["DEEPSEEK_API_KEY"] = api_key_input
    
    st.markdown("---")

    # File Uploader
    dm = DataManager()

with st.sidebar:
    st.title("🌤️ 气象分析仪")
    # ... (API Key 输入部分保持不变) ...
    
    st.markdown("---")
    st.markdown("### 📂 数据仓库")
    
    # 1. 上传新数据
    uploaded_file = st.file_uploader("上传新数据 (CSV/NetCDF)", type=['csv', 'txt', 'nc'])
    if uploaded_file:
        if dm.save_uploaded_file(uploaded_file):
            st.success(f"已归档: {uploaded_file.name}")
            # 重新加载页面以刷新列表
            st.rerun() 
            
    # 2. 选择历史数据
    stored_files = dm.list_files()
    if not stored_files.empty:
        selected_id = st.selectbox(
            "选择要分析的数据集",
            options=stored_files['id'].tolist(),
            format_func=lambda x: stored_files[stored_files['id'] == x]['filename'].values[0]
        )
        
        # 加载按钮
        if st.button("🚀 加载选中数据", use_container_width=True):
            file_path = dm.get_file_path(selected_id)
            with st.spinner("正在读取数据..."):
                try:
                    # 调用支持 nc 的新 loader
                    data = load_data(file_path)
                    st.session_state.df = data # 这里变量名建议改为 st.session_state.data 以避免混淆
                    
                    # 显示加载信息
                    if isinstance(data, pd.DataFrame):
                        st.success(f"CSV 加载成功: {len(data)} 行")
                    else:
                        st.success(f"NetCDF 加载成功: {str(data.dims)}")
                except Exception as e:
                    st.error(f"加载失败: {e}")
    else:
        st.info("仓库为空，请上传数据。")

# Initialize Agent (Re-initialize if API Key changes or first run)
# Note: st.cache_resource is good, but we need to invalidate it if API key changes.
# For simplicity, let's create a wrapper that checks the key.
@st.cache_resource(show_spinner=False)
def get_agent_instance(api_key_value):
    # api_key_signature is just to force recompilation when key changes
    return MeteorologyAgent(api_key=api_key_value)

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
                        data=st.session_state.df, 
                        history=st.session_state.messages
                    )
                    
                   # --- [修改代码开始] 适配 Code Interpreter ---
                    code = response.get("code", "")
                    result = response.get("result", "")
                    figure = response.get("figure", None)
                    new_df = response.get("new_df", None)
                    thought = response.get("thought", "") # 新版可能也会保留简单的 thought
                    action = response.get("action", "Code Execution")

                    if code:
                        st.markdown("### 💻 生成的代码")
                        st.code(code, language="python")
                
                    if new_df is not None and not new_df.empty:
                        st.session_state.df = new_df
                        st.toast("✅ 数据集已更新")
                    
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
                    # 如果是 Plotly 对象 -> 交互式展示
                    if isinstance(figure, (go.Figure,)):
                        st.plotly_chart(figure, use_container_width=True)
                    # 如果是 Matplotlib 对象 -> 静态展示
                    elif isinstance(figure, plt.Figure):
                        st.pyplot(figure)
                    else:
                        st.warning("生成了无法识别的图表对象。")
                
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
