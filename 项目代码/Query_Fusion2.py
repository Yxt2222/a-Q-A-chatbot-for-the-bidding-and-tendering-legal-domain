'''检索前优化模块
在检索前优化模块中，首先对用户传入的查询（Query）
进行停用词过滤，以减少噪声并提高检索的精确度。
然后，系统会基于优化后的查询生成多个衍生问题。
为了实现这一点，我们引入了一种微调的大型生成模型
（模型类型待定），其任务是根据用户的初始查询生成不同类型的问题。
为了增强生成问题的多样性和针对性，我们将问题类型划分为三类：
细节性问题、解释性问题和推理性问题。同时，通过使用提示词（prompts）和
小样本提示学习（few-shot learning），
模型能够学习如何生成这些不同类型的问题。'''

import hanlp
from sparkllm import ChatSparkLLM
import os
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate

# 停用词路径常量
STOPWORDS_PATH = r"C:\Users\86131\Desktop\科大讯飞实习\Stopwords-main\stopwords_cn.txt"

def load_stopwords(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

def lemmatize_chinese(text,stopwords):
    """
    中文词形还原函数，输入为一段中文文本，输出为还原后的文本。
    """
    # 初始化 HanLP
    tokenizer = hanlp.load('PKU_NAME_MERGED_SIX_MONTHS_CONVSEG')
    segments = tokenizer(text)
    lemmatized_words = []
    for segment in segments:
        if isinstance(segment, tuple) and len(segment) == 2:
            word, pos = segment
            if word not in stopwords:  # 去除停用词
                lemmatized_words.append(word)
        else:
            if segment not in stopwords:
                lemmatized_words.append(segment)  # 如果segment是单个值，则直接添加
    return ''.join(lemmatized_words)

PROMPT_TEMPLATE = '''
#01 你是一个招投标行业的法律专家
#02 你的任务是根据我给出的与法律相关的文本，生成若五个相关问题。问题必须与我提供的法律文本高度相关，不能仅仅凭借自己已有的知识生成问题。
#03 问题要尽量短，不要太长。
#04 生成的五个问题可以是细节性问题，理解性问题或推理性问题。
五个问题中两个是细节性问题，两个是理解性问题，一个是逻辑性问题。

细节性问题：针对数量，日期，金额，条件，惩罚，对象等要素提问。该问题的答案是输入文本的局部的具体细节。
理解性问题：针对定义，流程等要素进行提问。该问题的答案是与输入文本有关的关键词的直接解释或者阐述。
推理性问题：针对原因，结果等要素进行提问。该问题的答案不会由输入文本直接给出，而是由输入文本的部分经过推理，延申或者倒推得到。

#05细节性问题，理解性问题，推理性问题的示例：

输入文本：如果中标人想要将中标项目的部分工作分包给他人，需要满足什么条件？
细节性问题：标人将中标项目的部分工作分包给他人时，需要满足哪些具体条件？
理解性问题：中标人将部分工作分包给他人是否需要事先经过招标人同意？
推理性问题：如果中标人未满足条件就将部分工作分包，会导致什么法律后果？

输入文本：如果投标人以欺骗手段骗取中标，将会面临哪些法律责任？  
细节性问题：投标人以欺骗手段骗取中标时将会面临哪些具体的法律处罚？
理解性问题：在法律中，投标人以欺骗手段骗取中标是否被视为严重违法行为？
推理性问题：如果投标人以欺骗手段骗取中标，可能对其他投标人和招标人造成什么影响？

#06 生成的问题以'？'结尾前方不要带编号如’1，2，3，4‘。输出的每一行只有一个问句，不包含其他任何字符。你必须按照我给定的格式生成问题，指定格式为：
格式示例如下：

标人将中标项目的部分工作分包给他人时，需要满足哪些具体条件？
什么情况下，标人才能将中标项目的部分工作分包给他人？
中标人将部分工作分包给他人是否需要事先经过招标人同意？
招标人不同意时，中标人能否将部分工作分包给他人？
如果中标人未满足条件就将部分工作分包，会导致什么法律后果？
......
#07 所有回答必须使用中文，禁止使用英文。

#08 以下是我给出的内容：

"""
Content:{Content}

"""
'''

def query_fusion(query):
    '''
    query:str 去除停用词的查询
    api_url:str 大模型接口地址
    llm_domain: str 大模型domain参数
    patch_id:str 大模型resourceID参数
    '''
    stopwords = load_stopwords(STOPWORDS_PATH)
    context = lemmatize_chinese(query, stopwords)
    
    prompt =  PROMPT_TEMPLATE.format(Content=context)
    '''
    llm = SparkLLM(
        spark_api_url=api_url,
        spark_llm_domain= llm_domain,
        spark_patch_id=patch_id,
        model_kwargs={"temperature": 0.1}
    )'''
    llm= ChatSparkLLM(spark_app_id ='7bc5792a',
                    spark_api_key='9b23b496c1cd14e7ddeda626683c1e8b',
                    spark_api_secret='MmI1NDE5NzQ2M2QwMmQ5NWNjNTY1NzAy',
                    spark_api_url='wss://spark-api-n.xf-yun.com/v1.1/chat',
                    spark_llm_domain='patch',
                    spark_patch_id ='1826586836706508800',
                    temperature=0.1)

   
    response = llm.invoke(prompt)
    result =  response.content.split('\n')
    result.append(query)
    return result

