#%%
# 실제 chatbot을 만들어 보자! [Memory module] ***
## 강의의 모든 내용은 langchain의 튜토리얼을 참조해서 진행함. ***
from ModelLoad import ModelLoader
llm = ModelLoader.get_llm()
emb = ModelLoader.get_embedding()
tok = ModelLoader.get_tokenizer()
#%%
# Quick start에서 봤듯이 아래와 같이 각 메시지 별 객체를 생성하여 chatbot 구현이 가능함.
from langchain.schema import AIMessage, HumanMessage, SystemMessage
messages = [
    HumanMessage(
        content = "이 문장을 영어에서 한국어로 번역하세요 : I love programming."
    )
]
llm.invoke(messages)
#%%
messages = [
    SystemMessage(
        content = "당신은 영어에서 한국어로 번역하는 도움이 되는 조수입니다."
    ),
    HumanMessage(content = "I love programming")
]
llm.invoke(messages)

#%%
# ConversationChain
# ConversationChain은 유저의 입력과 모델의 출력 값을 history로 가지고 있는 built-in memory chain 모듈을 뜻함.
from langchain.chains import ConversationChain
conversation = ConversationChain(llm = llm)
conversation.invoke("이 문장을 영어에서 한국어로 번역하세요 : I love programming, 번역 내용만 출력하세요.")
#%%
# 위의 내용이 저장된 상태이므로 아래와 같이 추가 질문을 통해 기억하고 있는 지 확인.
conversation.invoke("독일어로 번역하세요.")
#%%
# 최근에는 Conversation을 사용하지 않고, RunnableWithMessageHistory를 응용하여 사용한다고 함.
# Conversation은 후 버전에서는 지원하지 않는다는 경고가 발생함.
## 아래는 LangChain에서 RunnableWithMessageHistory 사용 예시에 대한 내용을 가지고 옴.
from typing import List
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from langchain_core.runnables import (
    RunnableLambda,
    ConfigurableFieldSpec,
    RunnablePassthrough,
)
from langchain_core.runnables.history import RunnableWithMessageHistory

class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []

# Here we use a global variable to store the chat message history.
# This will make it easier to inspect it to see the underlying results.
store = {}

def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]


history = get_by_session_id("1")
history.add_message(AIMessage(content="hello"))
print(store)  # noqa: T201
#%%
# 아래 방식은 위의 get_by_session_id()를 이용하여, 단일 session_id를 받아 해당 세션의 대화 이력을 반환.
# session_id는 대화 세션을 구분하는 유일한 식별자로 사용됨.
## 호출 방식은 체인을 호출할 때, config 매개변수에 session_id를 지정함.
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

prompt = ChatPromptTemplate.from_messages([
    ("system", "You're an assistant who's good at {ability}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

chain = prompt | llm

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_by_session_id,
    input_messages_key="question",
    history_messages_key="history", # 위 prompt의 placeholder의 variable_name 참조.
)

print("Answer: ", chain_with_history.invoke(  # noqa: T201
    {"ability": "math", "question": "What does cosine mean?"},
    config={"configurable": {"session_id": "foo"}}
), "\n")

# Uses the store defined in the example above.
print("Memory infomation: \n")
print(store["foo"], "\n")  # noqa: T201

print("Answer: ", chain_with_history.invoke(  # noqa: T201
    {"ability": "math", "question": "What's its inverse"},
    config={"configurable": {"session_id": "foo"}}
), "\n")

print("Memory infomation: \n")
print(store["foo"], "\n")  # noqa: T201
# store의 "foo" key에 저장이 되어있음. (session_id 별로 저장이 가능한 것으로 보임)
#%%
# 아래는 get_session_history 함수를 이용하여 user_id와 conversation_id 두 개의 배개 변수를 받아 해당 사용자와 대화의 이력을 반환.
# 이를 통해 동일한 사용자의 여러 대화를 구분하여 관리가 가능.
## 호출 방식은 체인을 호출할 때, config 배개변수에 user_id와 conversation_id를 함께 지정.
# 
store = {} 

def get_session_history(
    user_id: str, conversation_id: str
) -> BaseChatMessageHistory:
    if (user_id, conversation_id) not in store:
        store[(user_id, conversation_id)] = InMemoryHistory()
    return store[(user_id, conversation_id)]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You're an assistant who's good at {ability}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

chain = prompt | llm

with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history=get_session_history,
    input_messages_key="question", # 위 prompt의 placeholder의 variable_name 참조.
    history_messages_key="history", # input 및 history message key는 변경 가능. 
    history_factory_config=[
        ConfigurableFieldSpec(
            id="user_id",
            annotation=str,
            name="User ID",
            description="Unique identifier for the user.",
            default="",
            is_shared=True,
        ),
        ConfigurableFieldSpec(
            id="conversation_id",
            annotation=str,
            name="Conversation ID",
            description="Unique identifier for the conversation.",
            default="",
            is_shared=True,
        ),
    ],
)

with_message_history.invoke(
    {"ability": "math", "question": "What does cosine mean?"},
    config={"configurable": {"user_id": "123", "conversation_id": "1"}}
)
# store[(user_id, conversation_id)] 이런 식으로 가지고 올 수 있음.
#%%
# 따로 실험을 해보기 위한 구현.
store = {} 

def get_session_history(
    user_id: str, conversation_id: str
) -> BaseChatMessageHistory:
    if (user_id, conversation_id) not in store:
        store[(user_id, conversation_id)] = InMemoryHistory()
    return store[(user_id, conversation_id)]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You're a good assistant"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

chain = prompt | llm
test_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history=get_session_history,
    history_messages_key="history",
    input_messages_key="question",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="user_id",
            annotation=str,
            name="User ID",
            description="Unique identifier for the user.",
            default="",
            is_shared=True,
        ),
        ConfigurableFieldSpec(
            id="conversation_id",
            annotation=str,
            name="Conversation ID",
            description="Unique identifier for the conversation.",
            default="",
            is_shared=True,
        ),
    ],
)
def chat(message_history, message, user_id, conversation_id):
    # MessageHistory를 제대로 사용하려면 prompt를 ChatPromptTemplate 형태로 작성해주는 것이 별다른 오류 발생 x.
    answer = message_history.invoke(
        {"question":f"{message}"},
        config={"configurable": {"user_id": user_id, "conversation_id": conversation_id}}
    )
    return answer

# 아래 코드를 통해 memory가 잘 작동함을 확인함.
user_id = "rwkim"
conversation_id = "test"
q1 = "제가 좋아하는 색상은 파란색입니다. 기억 해주세요."
answer = chat(test_message_history, q1, user_id, conversation_id)

print("===========================================================")
print("Memory information: ")
for message in store[user_id, conversation_id].messages: 
    print(message.content)
print("===========================================================")
q2 = "제가 좋아하는 색상은 무엇인가요?"
answer = chat(test_message_history, q2, user_id, conversation_id)
print("===========================================================")
print("Memory information: ")
for message in store[user_id, conversation_id].messages: 
    print(message.content)
print("===========================================================")
# %%
# 아래 코드를 통해 만일 store의 내용을 변경시킨다면 어떻게 될 지 확인.
store = {} 

def get_session_history(
    user_id: str, conversation_id: str
) -> BaseChatMessageHistory:
    if (user_id, conversation_id) not in store:
        store[(user_id, conversation_id)] = InMemoryHistory()
    return store[(user_id, conversation_id)]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You're a good assistant"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

chain = prompt | llm
test_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history=get_session_history,
    history_messages_key="history", 
    input_messages_key="question",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="user_id",
            annotation=str,
            name="User ID",
            description="Unique identifier for the user.",
            default="",
            is_shared=True,
        ),
        ConfigurableFieldSpec(
            id="conversation_id",
            annotation=str,
            name="Conversation ID",
            description="Unique identifier for the conversation.",
            default="",
            is_shared=True,
        ),
    ],
)

from langchain.schema import AIMessage, HumanMessage, SystemMessage
user_id = "rwkim"
conversation_id = "test2"
q1 = "제가 좋아하는 색상은 파란색입니다. 기억 해주세요."
answer = chat(test_message_history, q1, user_id, conversation_id)

print("===========================================================")
print("Memory information: ")
for message in store[user_id, conversation_id].messages: 
    print(message.content)
print("===========================================================")

del store[user_id, conversation_id].messages[0]
del store[user_id, conversation_id].messages[0]

store[user_id, conversation_id].messages.append(HumanMessage(content = "제가 좋아하는 색상은 빨간색입니다. 기억해주세요."))
store[user_id, conversation_id].messages.append(AIMessage(content = "예 기억하겠습니다."))

q2 = "제가 좋아하는 색상은 어떤 것이라고 말했었습니까?"
answer = chat(test_message_history, q2, user_id, conversation_id)
print("===========================================================")
print("Memory information: ")
for message in store[user_id, conversation_id].messages: 
    print(message.content)
print("===========================================================")
#%%
# 추가로 강의에서는 이전 대화 내역 요약 및 token 개수 제한 요약에 대한 내용을 다룸.
from langchain.memory import ConversationSummaryMemory
memory = ConversationSummaryMemory(llm=llm)
memory.save_context({"input": "안녕!"}, {"output": "잘지내?"})
memory.save_context({"input": "나는 챗봇을 위한 더 나은 문서 작업을 하고 있어"},
                    {"output": "오, 그거 꽤 많은 일처럼 들리네"})
memory.save_context({"input": "네, 하지만 그 노력이 가치가 있어요"},
                    {"output": "동의해, 좋은 문서는 중요해!"})
memory.load_memory_variables({})
#%%
# Token 수를 제한하여 요약한 이전 대화내역을 저장하는 코드
## 요약 이라는 특성 상 정확한 토큰 수 제한을 항상 엄격하게 준수하지 못할 수도 있음.
from langchain.memory import ConversationSummaryBufferMemory
memory = ConversationSummaryBufferMemory(llm = llm, max_token_limit=5)
memory.save_context({"input": "안녕!"}, {"output": "잘 지내?"})
memory.save_context({"input": "별로, 너는?"}, {"output": "별로"})
memory.load_memory_variables({})