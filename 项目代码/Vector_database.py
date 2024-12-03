'''
构建向量数据库模块：
首先，我们对准备好的、经过数据清洗的文本文件（如.txt或.md格式）进行文本切分。
这一过程通常结合递归字符划分、文档特定划分和语义划分等策略，
以最佳方式将文本分解成更小的语义单元（chunks）。
鉴于法律文件的高度规范性和结构性，在本项目中，我们优先选用了递归字符划分方法。
这种划分方式有助于保持文档的逻辑连贯性和法律条款的完整性。
接下来，使用微调后的'bge-base-zh-v1.5'嵌入模型将这些切分后的文本块（chunks）
转化为向量，并将这些向量存储在向量数据库中。向量数据库的构建是整个系统的核心，
它支持后续的高效检索与匹配过程。
'''
from langchain.schema import Document
import pickle
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from FlagEmbedding import FlagModel
from langchain_community.vectorstores import FAISS

class FlagModelEmbeddingWrapper:
    def __init__(self, flag_model):
        self.flag_model = flag_model

    def embed_documents(self, texts):
        # 替换 `encode` 为 `FlagModel` 中实际使用的嵌入方法
        return self.flag_model.encode(texts)
    
    def embed_query(self, query):
        # 为查询文本生成嵌入
        return self.flag_model.encode([query])[0]  # 假设 `encode` 方法返回列表
    
    def __call__(self, text):
        # 为了使对象可调用，默认调用 `embed_query`
        return self.embed_query(text)
    
class DocumentRetriever:
    def __init__(self, model_path, Chunk_size=150, Chunk_overlap=30):
        #中华人民共和国招标投标法律法规全书路径
        file_path1 = "C:/Users/86131/Desktop/科大讯飞实习/分享 8000个问答对全文件-罗鑫/中华人民共和国招标投标法律法规全书1.txt"
        # Load the documentc
        self.loader = TextLoader(file_path1, encoding='utf-8')
        self.docs = self.loader.load()

        # Split the document into chunks
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Chunk_size,
            chunk_overlap=Chunk_overlap,
        )
        texts1 = self.text_splitter.split_documents(self.docs)
        # 招标投标法律解读与风险防范实务切割文本路径
        file_path2  = "C:/Users/86131/Desktop/科大讯飞实习/分享 8000个问答对全文件-罗鑫/招标投标法律解读与风险防范实务切割文本.pkl"
        # 使用 pickle 加载招标投标法律解读与风险防范实务切割文本
        with open(file_path2, "rb") as file:
            text_chunks = pickle.load(file)
        # 将每个文本封装成 `Document` 对象
        texts2 = [Document(page_content=text) for text in text_chunks]
        #将第二份文档的划分文本段加载进去
        self.texts =  texts1 + texts2
        # Initialize the embedding model
        self.embedding_model = FlagModelEmbeddingWrapper(
            FlagModel(model_path, 
                      query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                      use_fp16=True)
        )

        # Create a vector database from the document chunks
        self.db = FAISS.from_documents(self.texts, self.embedding_model)
        self.retriever = self.db.as_retriever()

    def retrieve_top_k_single(self, query, k=3):
        '''
        query:用户输入的查询
        k:输出相关度最高的前k个文档
        '''
        """Retrieve top-k relevant chunks based on single query."""
        relevant_docs = self.retriever.get_relevant_documents(query, k=k)
        return relevant_docs
    
    def retrieve_top_k_multiple(self, query_list, k=3):
        '''
        query_list:若干查询组成的列表
        k:对每个查询，输出相关度最高的前k个文档
        '''
        """Retrieve top-k relevant chunks based on several queries."""
        retrieved_chunks = []
        for query in query_list:
            relevant_docs = self.retriever.get_relevant_documents(query, k=k)
            retrieved_chunks.append([doc.page_content for doc in relevant_docs])
        return retrieved_chunks

 