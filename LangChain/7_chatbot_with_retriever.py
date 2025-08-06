#%%
# 이번엔 retriever 기능을 이용하여 chatbot을 만들어 보자.
# 해당 섹션에서는 강의의 내용을 토대로 그대로 진행해보려고 함.
## 최근 추세는 Conversation(전 자동화) 보다는 약간의 custom이 가능한 Runnable 모듈 쪽으로 업데이트 방향이 잡힘.
## 해당 강의에서 llm + retriever + memory를 활용하는 conversation 모듈이 Runnable 모듈에는 존재 x. 10장에서 개인적으로 구현 진행 예정.
from ModelLoad import ModelLoader
llm = ModelLoader.get_llm()
emb = ModelLoader.get_embedding()
tok = ModelLoader.get_tokenizer()
#%%
# 앞선 섹션 6에서는 기존의 conversation을 제대로 사용하지 않았음.
# 그러므로 아래에서 간단히 memory를 사용한 예시를 본 후, llm+retriever+memory 진행.
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

# Prompt
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "너는 인간과 대화를 나누는 친절한 챗봇이야."
        ),
        # variable_name이 memory와 연결하는 key입니다.
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
conversation = LLMChain(llm=llm, prompt=prompt, verbose=True, memory=memory)

conversation({"question": "안녕"})
#%%
conversation(
    {"question": "이 문장을 영어에서 한국어로 번역하세요 : I love programming."}
)
#%%
conversation({"question": "독일어로 번역하세요."})
# model의 성능 문제인지, 더 이상 관리가 안되는 모듈이여서 그런지 모르겠지만, history 내역이 이상하게 남음.
#%%
# LLM + retriever + memory 활용 예시.
from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("https://snuac.snu.ac.kr/2015_snuac/wp-content/uploads/2015/07/asiabrief_3-26.pdf")
pages = loader.load_and_split()
pages[0]
#%%
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
splits = text_splitter.split_documents(pages)
splits
#%%
from langchain.vectorstores import Chroma
vectorstore = Chroma.from_documents(documents = splits, embedding = emb)
memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True
)
#%%
from langchain.chains import ConversationalRetrievalChain
retriever = vectorstore.as_retriever()
qa = ConversationalRetrievalChain.from_llm(llm, retriever = retriever, memory = memory)
qa("저출산을 극복한 나라들은 어디가 있어?")
#%%
qa("최신 관련 자료를 알려줘.")
#%%
qa("최신 관련 자료들의 문헌 제목을 알려줘.")