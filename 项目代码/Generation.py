'''
生成模块：
在生成模块中，经过前期调研与评估，我们选用了Spark Max作为生成大模型。
并且构建的带记忆的检索增强生成链是带记忆的。
结合用户的初始查询和经过优化的参考文本块，生成模型生成最终的回答。
'''


from langchain.memory import ConversationBufferMemory
from langchain_community.llms import SparkLLM
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
'''
os.environ["IFLYTEK_SPARK_APP_ID"] = "7bc5792a"
os.environ["IFLYTEK_SPARK_API_KEY"] = "9b23b496c1cd14e7ddeda626683c1e8b"
os.environ["IFLYTEK_SPARK_API_SECRET"] = "MmI1NDE5NzQ2M2QwMmQ5NWNjNTY1NzAy"
'''
prompt_template='''
你是一名专门回答招投标法律相关问题的智能助手。
你的任务是根据以下提供的专业文献片段，准确回答用户提出的问题。
如果你无法从提供的文献中找到答案，请直接告知用户你不知道。
请确保你的回答简明扼要，并且完全基于文献内容。
直接回答就行，不要输出类似‘根据你提供的内容’，‘根据提供的专业文献片段’等的内容。
下面标记为Memory,Context和Query的三栏分别表示历史对话信息，相关专业文献和用户的提问。
-------------------------------------------------------
Memory:{Memory}
-------------------------------------------------------
Context:{Context}
-------------------------------------------------------
Query:{Query}
'''

# Initialize conversation memory
#memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def generate_reply(query,reference,memory_content):
    #将context转化成字符串
    context = "\n".join(reference)
    #定义大模型
    llm = SparkLLM(
        spark_api_url='wss://spark-api.xf-yun.com/v3.5/chat',
        spark_llm_domain='generalv3.5',
        model_kwargs={"temperature": 0.1}
    )
    # 创建提示链
    simple_chat_chain = (PromptTemplate.from_template(prompt_template) | llm | StrOutputParser())
    # 检索并格式化记忆内容
    #memory_variables = memory.load_memory_variables({})
    #memory_content = memory_variables.get("chat_history", "")
    # 生成回复
    ai_message = simple_chat_chain.invoke({
        "Memory": memory_content,
        "Context": context,
        "Query": query
    })
    # 保存历史对话信息
    #memory.save_context({"query": query}, {"response": ai_message})
    return ai_message


 