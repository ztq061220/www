# 标准库导入
import os
import re
import time
import requests
from bs4 import BeautifulSoup

# LangChain 相关导入（适配 0.1.0 稳定版）
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# 读取 .env 文件
from dotenv import load_dotenv
load_dotenv()

# ---------------------- 【配置区：按你的实际情况修改】 ----------------------
# 福州大学官网基础URL（根据实际链接调整）
BASE_URL = "https://www.fzu.edu.cn"
# 要爬取的「学校概况」子页面（请替换为真实路径）
PAGES_TO_CRAWL = [
    ("学校简介", "/xxgk/xxjj.htm"),
    ("学校章程", "/xxgk/xxzc.htm"),
    ("现任领导", "/xxgk/xrld.htm"),
    ("院士风采", "/xxgk/ysfc.htm"),
    ("校标校训", "/xxgk/xxbx.htm")
]
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
VECTOR_DB_PATH = "fzu_vector_db"
# -------------------------------------------------------------------------

# 加载环境变量（OpenAI API Key）
load_dotenv()

# ====================== 1. 文本清洗函数 ======================
def clean_text(text):
    if not text:
        return ""
    # 去除多余空白
    text = re.sub(r'\s+', ' ', text).strip()
    # 去除特殊符号
    text = re.sub(r'[^\u4e00-\u9fff\w\s，。、；：？！《》（）“”‘’]', '', text)
    # 过滤过短内容
    if len(text) < 20:
        return ""
    return text

# ====================== 2. 爬虫：批量爬取学校概况页面 ======================
def crawl_fzu_pages():
    crawled_docs = []
    print("=== 开始爬取福州大学「学校概况」页面 ===")
    
    for page_name, url_path in PAGES_TO_CRAWL:
        full_url = BASE_URL + url_path
        print(f"爬取：{page_name} → {full_url}")
        try:
            resp = requests.get(full_url, headers=HEADERS, timeout=15)
            resp.encoding = "utf-8"
            soup = BeautifulSoup(resp.text, "lxml")
            
            # 移除无用标签
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
            
            # 提取正文（如果官网正文有固定class，可改为 soup.find("div", class_="xxx")）
            raw_text = soup.get_text()
            cleaned_text = clean_text(raw_text)
            
            if cleaned_text:
                crawled_docs.append(
                    Document(
                        page_content=cleaned_text,
                        metadata={"source": page_name}
                    )
                )
                print(f"✅ {page_name} 爬取完成")
            else:
                print(f"⚠️ {page_name} 未提取到有效内容")
            time.sleep(1)
        except Exception as e:
            print(f"❌ {page_name} 爬取失败：{e}")
    
    print(f"=== 爬取完成，共获取 {len(crawled_docs)} 个文档 ===")
    return crawled_docs

# ====================== 3. 构建向量库 ======================
def build_vector_store(documents):
    print("=== 构建向量库 ===")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    split_docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(split_docs, embeddings)
    vector_store.save_local(VECTOR_DB_PATH)
    print("✅ 向量库构建完成并保存")
    return vector_store

# ====================== 4. 初始化 Agentic RAG ======================
def init_agentic_rag(vector_store):
    print("=== 初始化 Agentic RAG ===")
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    def retrieve_info(query):
        docs = retriever.get_relevant_documents(query)
        return "\n".join([doc.page_content for doc in docs])

    tools = [
        Tool(
            name="福州大学知识库检索",
            func=retrieve_info,
            description="当你需要查询福州大学的相关信息时，必须使用此工具获取准确内容"
        )
    ]

    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是福州大学智能问答助手，回答问题必须基于检索到的官方信息，不要编造。
你有一个工具可以使用：
{tools}
使用工具的格式是：
Action: 工具名称
Action Input: 你的查询
"""),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor

# ====================== 5. 主程序 ======================
if __name__ == "__main__":
    # 步骤1：爬取数据（如果向量库已存在，可跳过爬取）
    if not os.path.exists(VECTOR_DB_PATH):
        crawled_docs = crawl_fzu_pages()
        if not crawled_docs:
            print("❌ 未爬取到有效数据，无法继续")
            exit()
        vector_store = build_vector_store(crawled_docs)
    else:
        print("=== 检测到已存在向量库，直接加载 ===")
        vector_store = FAISS.load_local(VECTOR_DB_PATH, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    
    # 步骤2：启动 Agentic RAG
    agent_executor = init_agentic_rag(vector_store)
    print("\n🤖 Agentic RAG 已启动！输入问题即可查询福州大学信息（输入 q 退出）")
    while True:
        user_query = input("你：")
        if user_query.lower() == "q":
            break
        result = agent_executor.invoke({"input": user_query})
        print(f"\n🤖 助手：{result['output']}\n")