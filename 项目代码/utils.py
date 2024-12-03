from Query_Fusion2 import *
from Vector_database import *
from Rerank_Module import *
from Generation import *


#嵌入模型路径
model  = "C:/大语言模型库/Finetuned_model/BAAI/bge-base-zh-v1.5"
#Rewrite模型的domain
llm_domain = ''
#Rewrite模型的patch_id
patch_id = ''

#构建向量数据库检索器
retriever = DocumentRetriever(model_path = model)
 

def get_response(Query,memory):
    # 检索当前的记忆内容
    memory_variables = memory.load_memory_variables({})
    memory_content = memory_variables['history']
    print(memory_content)
    #对输入的Query执行检索前优化，去除停用词，改写成多个问题。
    Fusion_query =  query_fusion(Query)
    #检索每个Fusion对应的relevant Chunks
    chunks_list = retriever.retrieve_top_k_multiple(Fusion_query,k = 5)
    #Rerank获取综合评分前3的chunk作为reference
    reference = Rerank(retrieved_chunks=chunks_list ,k=3)
    #根据query 和 reference合成prompt，喂给大模型，生成回答。
    ai_message = generate_reply(Query,reference,memory_content)
    # 保存当前对话信息到记忆
    memory.save_context({"query": Query}, {"response": ai_message})
    
    return ai_message,reference
 