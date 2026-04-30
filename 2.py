import os
import re
import time
import requests
import textwrap
import re
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.llms import Tongyi
from langchain.embeddings import DashScopeEmbeddings

# 去阿里云申请免费API Key：https://dashscope.console.aliyun.com/
os.environ["DASHSCOPE_API_KEY"] = "sk-d0eaa9a45b7d4f888f8ddbf62a28a6e4"

# ===================== 配置 =====================
BASE_URL = "https://www.fzu.edu.cn"
PAGES_TO_CRAWL = [
    ("学校简介", "/xxgk/xxjj.htm"),
    ("学校章程", "/xxgk/xxzc.htm"),
    ("现任领导", "/xxgk/xrld.htm"),
    ("院士风采", "/xxgk/ysfc.htm"),
    ("校标校训", "/xxgk/xxbx.htm")
]
VECTOR_DB_PATH = "fzu_vector_db"

# ===================== 文本清洗 =====================
def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\u4e00-\u9fff\w\s，。、；：？！《》（）“”‘’]', '', text)
    return text if len(text) > 20 else ""

# ===================== 爬虫 =====================
def crawl_fzu_pages():
    crawled_docs = []
    print("=== 开始爬取福州大学「学校概况」页面 ===")
    for page_name, url_path in PAGES_TO_CRAWL:
        full_url = BASE_URL + url_path
        print(f"爬取：{page_name} → {full_url}")
        try:
            resp = requests.get(full_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            resp.encoding = "utf-8"
            soup = BeautifulSoup(resp.text, "lxml")
            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()
            cleaned_text = clean_text(soup.get_text())
            if cleaned_text:
                crawled_docs.append(Document(page_content=cleaned_text, metadata={"source": page_name}))
                print(f"✅ {page_name} 爬取完成")
            time.sleep(1)
        except Exception as e:
            print(f"❌ {page_name} 爬取失败：{e}")
    print(f"=== 爬取完成，共获取 {len(crawled_docs)} 个文档 ===")
    return crawled_docs

# ===================== 构建向量库（通义千问嵌入） =====================
def build_vector_store(documents):
    print("=== 构建向量库 ===")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = text_splitter.split_documents(documents)
    embeddings = DashScopeEmbeddings(model="text-embedding-v1")
    vector_store = FAISS.from_documents(split_docs, embeddings)
    vector_store.save_local(VECTOR_DB_PATH)
    print("✅ 向量库构建完成")
    return vector_store

# ===================== 初始化Agent（通义千问大模型） =====================

def simple_chat(vector_store):
    print("\n🤖 福大智能助手已启动！输入 q 退出")
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})  # 放宽参数
    llm = Tongyi(model="qwen-turbo", temperature=0)

    while True:
        user_query = input("你：")
        if user_query.lower() == "q":
            break

        try:
            # 加 try-except，捕获检索失败的情况
            docs = retriever.get_relevant_documents(user_query)
            if not docs:
                context = "暂无相关官方信息。"
            else:
                context = "\n".join([doc.page_content for doc in docs])

            prompt = f"""
你是福州大学官方智能助手，回答必须严格基于下面的官方信息，不许编造。
回答格式：
1. 开头先给一句简短的结论；
2. 正文用普通段落叙述，不要用任何特殊符号；
3. 信息不完整时，直接说明“暂无公开信息”。

官方信息：
{context}

用户问题：{user_query}
"""
            response = llm.invoke(prompt)
            clean_response = re.sub(r'^-\s*', '', response, flags=re.MULTILINE).strip()

            print(f"\n🤖 助手：")
            print("-" * 60)
            print(textwrap.fill(clean_response, width=60))
            print("-" * 60)
            print()

        except Exception as e:
            print(f"\n⚠️ 检索出错了：{e}")
            print("🤖 助手：暂无相关官方信息。\n")

        
    
if __name__ == "__main__":
    if not os.path.exists(VECTOR_DB_PATH):
        crawled_docs = crawl_fzu_pages()
        if not crawled_docs:
            print("❌ 未爬取到有效数据，无法继续")
            exit()
        vector_store = build_vector_store(crawled_docs)
    else:
        print("=== 检测到已存在向量库，直接加载 ===")
        embeddings = DashScopeEmbeddings(model="text-embedding-v1")
        vector_store = FAISS.load_local(VECTOR_DB_PATH, embeddings)

    # 直接调用 simple_chat，不用 Agent
    simple_chat(vector_store)