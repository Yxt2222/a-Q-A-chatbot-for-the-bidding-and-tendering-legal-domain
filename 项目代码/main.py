import streamlit as st
from langchain.memory import ConversationBufferMemory
from utils import get_response
import os

# 设置页面配置
st.set_page_config(page_title="招投标法律行业问答机器人", page_icon="💬", layout="wide")

# 自定义CSS样式
st.markdown("""
    <style>
    /* 调整全局页边距 */
    .css-18e3th9 {padding: 1rem 2rem;}

    /* 修改字体 */
    .css-1l02zno {font-family: 'Roboto', sans-serif;}

    /* 美化按钮 */
    .stButton>button {border-radius: 12px; background-color: #007BFF; color: white;}
    .stButton>button:hover {background-color: #0056b3;}

    /* 美化扩展框 */
    .st-expander {border-radius: 12px; background-color: #f0f2f6; box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.1);}

    /* 美化对话框 */
    .st-chat-message {border-radius: 12px; background-color: #f8f9fa; box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.1); padding: 1rem;}

    /* 让图片在左边栏居中 */
    .stImage {display: flex; justify-content: center;}

    /* 让问答区域在页面中居中 */
    .main-content {display: flex; justify-content: center;}
    </style>
    """, unsafe_allow_html=True)

 



st.title('💬 招投标法律行业问答机器人')
 
with st.sidebar:
    #显示公司logo
    st.image(r"C:\Users\86131\Desktop\科大讯飞实习\科大讯飞logo.png", width=280)

    IFLYTEK_SPARK_APP_ID = st.text_input("请输入IFLYTEK_SPARK_APP_ID：", type="password")
    IFLYTEK_SPARK_API_KEY = st.text_input("请输入IFLYTEK_SPARK_API_KEY：", type="password")
    IFLYTEK_SPARK_API_SECRET = st.text_input("请输入IFLYTEK_SPARK_API_SECRET：", type="password")
    st.markdown("[获取Spark 大模型使用权](https://xinghuo.xfyun.cn/spark)")

    # 清空对话按钮，添加垃圾桶图标
    if st.button("🗑 清空对话"):
        st.session_state["messages"] = [{"role": "ai", "content": "你好，我是你的招投标法律助手，有什么可以帮你的吗？"}]
        st.session_state["memory"].clear()

    # 创建一个容器，容器中只放图片
     

os.environ["IFLYTEK_SPARK_APP_ID"] = IFLYTEK_SPARK_APP_ID
os.environ["IFLYTEK_SPARK_API_KEY"] = IFLYTEK_SPARK_API_KEY
os.environ["IFLYTEK_SPARK_API_SECRET"] = IFLYTEK_SPARK_API_SECRET


if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(return_messages=True)
    st.session_state["messages"] = [{"role": "ai",
                                     "content": "你好，我是你的招投标法律助手，有什么可以帮你的吗？"}]
#显示所有对话信息
for message in st.session_state["messages"]:
    st.chat_message(message["role"]).write(message["content"])

#用户输入信息
prompt = st.chat_input()
if prompt:
    if not IFLYTEK_SPARK_APP_ID:
        st.info("请输入你的IFLYTEK_SPARK_APP_ID")
        st.stop()
    if not IFLYTEK_SPARK_API_KEY:
        st.info("请输入你的IFLYTEK_SPARK_API_KEY")
        st.stop()
    if not IFLYTEK_SPARK_API_SECRET:
        st.info("请输入你的IFLYTEK_SPARK_API_SECRET")
        st.stop()   

    #将用户信息加入session_state
    st.session_state["messages"].append({"role": "human", "content": prompt})

    #显示用户问答信息
    st.chat_message("human").write(prompt)

    with st.spinner("AI正在思考中，请稍等..."):
        #获得ai回复
        response,reference = get_response(prompt, st.session_state["memory"])

    #拼接相关文档   
    reference_context = "\n".join(reference)

    #将ai回复加入session_state 
    msg = {"role": "ai", "content": response}
    st.session_state["messages"].append(msg)

    #显示ai回复
    st.chat_message("ai").write(response)

    # 显示可折叠的参考文本
    with st.expander("点击展开查看参考文本"):
        st.markdown(reference_context)

