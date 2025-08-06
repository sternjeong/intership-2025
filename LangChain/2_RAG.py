#%%
from ModelLoad import ModelLoader
llm = ModelLoader.get_llm()
#%%
# pip install pyparsing pytz pypdf 설치 (langchain_community 설치 위함.)
# pip install langchain_community 설치 필수
# Data connection은 Source -> Load -> Transform -> Embed -> Store -> Retrieve 순서로 진행.

# PDF loader에는 아래와 같은 다양한 종류가 존재 강의에서는 PyPDFLoader가 사용됨.
## 1. PyPDFLoader: 텍스트 추출에 특화되어 있으며, 표나 이미지를 직접 추출하는 기능은 제한적.
## 2. PDFPlumberLoader: PDF에서 텍스트, 이미지, 표 등 다양한 요소를 추출할 수 있는 강력한 도구. (특히 복잡한 구조의 표를 정확히 파싱)
## 3. PyMuPDFLoader(Fitz기반): 텍스트 뿐 아니라 이미지 주석 등의 정보를 추출하는 데 뛰어난 성능을 보여줌. (페이지 단위로 접근이 가능)
## 4. UnstructuredPDFLoader: 비정형 데이터 처리를 위한 라이브러리로, 다양한 파일 셩식의 구조 분석 및 텍스트를 효율적으로 추출 (레이아웃이 없는 PDF에서도 텍스트 추출 가능)
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("https://snuac.snu.ac.kr/2015_snuac/wp-content/uploads/2015/07/asiabrief_3-26.pdf")
# PDF를 불러와 적절한 크기로 분할.
## Data connection 과정의 Load 부분에 해당함. (split 아님)
pages = loader.load_and_split()
print(pages[0])

#%%
# Page 수 확인
print("Number of pages: ", len(pages))

#%%
# Data connection 과정의 transformer 부분에 해당함. (split 과정)
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
splits = text_splitter.split_documents(pages)
print("Number of splits: ", len(splits))
print(splits[0].page_content) # 한 split 내의 text 내용
print(splits[0].metadata) # Text 내용을 제외한 metadata들

#%%
# Embed, Stores, Retrieve 부분에 해당함.
## Vector stores 및 Embedding model 정의.
## Vector stores의 경우, 강의에서는 Chroma 사용.
### Chroma 사용을 위해 pip install chromadb 필수.
from langchain.vectorstores import Chroma
emb_model = ModelLoader.get_embedding()
vector_store = Chroma.from_documents(documents=splits, embedding = emb_model)
# vector store에 저장된 데이터들 중, question emb과 가장 유사한 emb vector를 찾아주는 retriever load.
retriever = vector_store.as_retriever()

#%%
# RAG와 LLM을 이용한 예제는 아래와 같음.
## 우선 template을 생성 (prompt engineering)
from langchain.prompts import PromptTemplate

# 아래 template의 context에는 vector store에서 찾은 유사한 정보가 들어감.
## 추가로 RAG를 사용하지 않았을 때와 사용했을 때의 답변 차이를 알기 위한 예시들도 새로 만듦.
rag_template = """다음과 같은 맥락을 사용하여 마지막 질문에 대답 해주세요.
만일 답을 모르면 모른다고만 말하고 답을 지어내려고 하지 마세요.
답변은 최대 세 문장으로 하고 가능한 한 간결하게 유지 해주세요.
{context}
질문: {question}
답변: 
"""
no_rag_template = """마지막 질문에 대답 해주세요.
만일 답을 모르면 모른다고만 말하고 답을 지어내려고 하지 마세요.
답변은 최대 세 문장으로 하고 가능한 한 간결하게 유지 해주세요.
질문: {question}
답변: 
"""

rag_template = PromptTemplate.from_template(rag_template)
no_rag_template = PromptTemplate.from_template(no_rag_template)

# RAG chain을 설정. 각 요소들에 대한 간단한 설명은 아래와 같음.
## retriever: 입력받은 question에 대해 관련된 context를 검색하여 반환.
## RunnablePassthrough(): 입력받은 질문을 그대로 전달하는 특수한 컴포넌트 (데이터 변형 x).
## RunnablePassthrough()를 사용하면, 사용자 입력이 특별한 처리 없이 그대로 입력으로 들어가지는 것으로 보임.
from langchain.schema.runnable import RunnablePassthrough

rag_chain = {"context": retriever, "question": RunnablePassthrough()} | rag_template | llm
no_rag_chain = no_rag_template | llm

#%%
# 예시 질문 1
question1 = "한국 저출산의 원인이 무엇입니까?"
rag_answer = rag_chain.invoke(question1)
no_rag_answer = no_rag_chain.invoke(question1)

if "\n" in rag_answer: 
    print(f"Question: {question1} \nAnswer(RAG): {rag_answer} \nAnswer(no RAG): {no_rag_answer}")
else: 
    # 깔끔하게 출력을 보기 위해서 추가함.
    rag_answer = '. \n'.join(rag_answer.split('. '))
    if "\n" not in no_rag_answer: 
        no_rag_answer = '. \n'.join(no_rag_answer.split('. '))
    print(f"Question: {question1} \nAnswer(RAG): {rag_answer} \nAnswer(no RAG): {no_rag_answer}")
    
#%%
# 예시 질문 2
question2 = "저출산을 극복한 나라들은 어디가 있어요?"
rag_answer = rag_chain.invoke(question2)
no_rag_answer = no_rag_chain.invoke(question2)
print(f"Question: {question1} \nAnswer(RAG): {rag_answer} \nAnswer(no RAG): {no_rag_answer}")

#%%
# 예시 질문 3
## RAG를 활용한 chain의 경우, 입력한 데이터에 없으면 모른다고 잘 대답함.
question3 = "대한민국의 강남구에 살고 있는 사람은 몇 명쯤 일까요?"
rag_answer = rag_chain.invoke(question3)
no_rag_answer = no_rag_chain.invoke(question3)
print(f"Question: {question1} \nAnswer(RAG): {rag_answer} \nAnswer(no RAG): {no_rag_answer}")
