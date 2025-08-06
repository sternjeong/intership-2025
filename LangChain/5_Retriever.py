#%%
# 이제 실제 vector store의 retriever를 응용해보자.
from langchain.document_loaders import PyPDFLoader
from ModelLoad import ModelLoader
llm = ModelLoader.get_llm()
emb = ModelLoader.get_embedding()
tok = ModelLoader.get_tokenizer()

# 1차 적으로 분할.
loader = PyPDFLoader("https://snuac.snu.ac.kr/2015_snuac/wp-content/uploads/2015/07/asiabrief_3-26.pdf")
pages = loader.load_and_split()
pages[0]
#%%
# PDF 내용을 더 작은 chunk 단위로 나누기
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
splits = text_splitter.split_documents(pages)
splits
#%%
# vector store 및 retriever 호출
## 여기서 retriever은 우선 LLM 연결 없이, 순수 Chroma의 embedding vector store만 이용.
from langchain.vectorstores import Chroma
vectorstore = Chroma.from_documents(documents = splits, embedding = emb)
retriever = vectorstore.as_retriever()
#%%
# LLM이 없이 순수 embedding 유사도만을 가지고 진행했을 때의 예시.
# 사람이 보기에는 가독성이 좋지 않음.
retrieved_docs = retriever.invoke(
    "저출산을 극복한 나라들은 어디가 있어?"
)
print(retrieved_docs[0].page_content)

#%%
# LLM을 이용하여 최종 데이터를 추출하려면 chain을 이용.
from langchain.prompts import PromptTemplate
template = """다음과 같은 맥락을 사용하여 마지막 질문에 대답하십시오.
만약 답을 모르면 모른다고만 말하고 답을 지어내려고 하지 마십시오.
답변은 최대 세 문장으로 하고, 가능한 한 간결하게 유지하십시오.
{context}
질문: {question}
도움이 되는 답변: """
rag_prompt_custom = PromptTemplate.from_template(template)
#%%
from langchain.schema.runnable import RunnablePassthrough
# 위에서 추출된 retriever 값들이 입력됨.
rag_chain = {"context": retriever, "question": RunnablePassthrough()} | rag_prompt_custom | llm

#%%
rag_chain.invoke("저출산을 극복한 나라들은 어디가 있어?")