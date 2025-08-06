#%%
# 앞서 배웠던 지식들을 바탕으로 여러 버전의 chatbot을 생성해 봄.
from ModelLoad import ModelLoader
llm = ModelLoader.get_llm()
emb = ModelLoader.get_embedding()
tok = ModelLoader.get_tokenizer()
#%%
# langchain의 message 객체를 list에 추가해가며 작동하는 chatbot.
from langchain.schema import SystemMessage, HumanMessage, AIMessage
use_system_message = input("System message setting: (yes or no)")
if use_system_message == "yes": 
    system_message = input("Enter system message (모델에게 역할을 부여): ")
    messages = [SystemMessage(content = system_message)]
else: 
    messages = []
    
c = 1
while True: 
    user_prompt = input("질문을 해주세요. (종료하려면 'end')")
    if user_prompt == 'end': 
        break
    messages.append(HumanMessage(content = user_prompt))
    answer = llm.invoke(messages)
    print(f"""User question {c}: {user_prompt} \n LLM Answer: {''.join(answer.split(':')[1:])[1:]} \n""", flush=True)
    messages.append(AIMessage(content = "".join(answer.split(":")[1:])[1:]))
    c+=1
#%%
# History를 좀 더 효율적이고, custom하게 진행하며 작동하는 chatbot.
## ChatPromptTemplate 활용.
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from typing import List
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, AIMessage
from pydantic import BaseModel, Field
from langchain_core.runnables import (
    RunnableLambda,
    ConfigurableFieldSpec,
    RunnablePassthrough,
)

use_system_message = input("System message setting: (yes or no)")
if use_system_message == "yes": 
    system_message = input("Enter system message (모델에게 역할을 부여): ")
    prompt = ChatPromptTemplate.from_messages([
    ("system", f"{system_message}"),
    MessagesPlaceholder(variable_name="history", n_messages = 100), # 삽입될 메시지 목록을 나타내는 변수명, n_messages를 통해 삽입할 최대 메시지 수를 지정할 수 있음.
    ("human", "{question}"),
    ])
else: 
    prompt = ChatPromptTemplate.from_messages([
    ("system", ""),
    MessagesPlaceholder(variable_name="history", n_messages = 100),
    ("human", "{question}"),
    ])

class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""
    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []

histories = {}

def get_session_history(
    user_id: str, conversation_id: str
) -> BaseChatMessageHistory:
    if (user_id, conversation_id) not in histories:
        histories[(user_id, conversation_id)] = InMemoryHistory()
    return histories[(user_id, conversation_id)]

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
#%%
# 메모리를 아낄 수 있는 방법에 대한 구상 (rwkim)
## 아래 실습 코드에서는 사용하지 않겠지만, 추후 각각의 answer들을 vectorstore에 저장하여 필요할 때마다 꺼내 쓸 수도 있을 것으로 보임.
from langchain.vectorstores import Chroma
vectorstore = Chroma.from_texts(texts = [""], embedding = emb)

## 아래와 같이 llm 대답 이후, summary_template을 사용하여 history를 요약본으로 교체.
summary_template = """다음과 같은 맥락을 사용하여 마지막 질문에 대답 해주세요.
만일 답을 모르면 모른다고만 말하고 답을 지어내려고 하지 마세요.
답변은 최대 세 문장으로 하고 가능한 한 간결하게 유지 해주세요.
{context}
질문: {question}
답변: 
"""
example_summary_answer = "answer~~"
### history 최신 대답을 지운 후 (AIMessage, 즉 answer 데이터 지우기), 
del histories["rwkim", "user"].messages[-1]
histories["rwkim", "user"].messages.append(AIMessage(content = example_summary_answer))


#%%
user_id = "rwkim"
conversation_id = "test"
while True: 
    question = input("질문을 해주세요. (종료하려면 'end' 입력)")
    if question == 'end': 
        break
    answer = with_message_history.invoke(
            {"question": question},
            config={"configurable": {"user_id": user_id, "conversation_id": conversation_id}}
            )
    print(f"Answer: {answer}", flush=True)
    check_history = input("이전 대화 내역을 확인 해보시겠습니까? (yes or no)")
    if check_history == "yes": 
        num_q = 1
        for idx, history in enumerate(histories[user_id, conversation_id].messages): 
            if type(history) == HumanMessage: 
                print(f"Chat sequence {num_q} ====================", flush=True)
                print(f"User question: {history.content}", flush=True)
                print(f"Answer: {histories[user_id, conversation_id].messages[idx+1].content}", flush=True)
                num_q += 1
                
                
    