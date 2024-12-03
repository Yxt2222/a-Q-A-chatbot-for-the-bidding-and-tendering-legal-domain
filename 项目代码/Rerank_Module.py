'''
检索后优化：
在得到一系列派生问题后，系统将这些问题向量化，并在向量数据库中检索相关的文本块
（chunks），生成一个排序后的相关文本块集合。在检索后优化模块中，
我们采用了倒数排序融合算法（Reciprocal Rank Fusion，RRF）
对这些相关文本块进行重排序（rerank）。RRF算法通过整合不同问题生成的多个文本块的排名，
计算综合得分（RRF score），得分越高的文本块与用户查询的相关性越强。
最终，我们选取得分最高的前 k 个文本块作为参考文档（reference），
并与用户的查询一同合成提示词，输入到生成模块中。
'''

from collections import defaultdict
import numpy as np

# Function to apply Reciprocal Rank Fusion (RRF)
def reciprocal_rank_fusion(ranked_lists, rank_constant=20,weight_constant=0.8):
    '''
    ranked_lists:[[str]]包含每个查询检索出的chunks列表的列表
    rank_constant：int rrf常数
    weight_constant:权重常数，float（0-1），weight_constant越大，原query的权重就越大，改写query的权重就越小
    '''
    # 创建一个默认值为浮点数的字典，用于存储每个文档的分数
    scores = defaultdict(float)
    weights = [(1-weight_constant)/(len(ranked_lists)-1) for _ in range(len(ranked_lists)-1)]
    weights.append(weight_constant)
    # 遍历每个排名列表
    for i in range(len(ranked_lists)):
        # 遍历当前排名列表中的每个文档及其排名
        rank_list = ranked_lists[i]
        for rank, doc in enumerate(rank_list):
            # 计算文档的 RRF 分数并累加到总分数中
            scores[doc] += weights[i] / (rank + 1.0 + rank_constant)
    # 将文档按分数从高到低排序，并返回排序后的结果
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# 定义函数，用于重新排序检索到的文本块
def Rerank(retrieved_chunks,k=1,rank_constant=20):
     # 使用 RRF 算法对检索到的文本块进行重新排序
    reranked_chunks = reciprocal_rank_fusion(retrieved_chunks,rank_constant=rank_constant)
    # 从重新排序的结果中提取前 k 个文本块
    reference = [reranked_chunks[i][0] for i in range(k)]
    return reference

 