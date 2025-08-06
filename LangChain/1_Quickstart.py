#%%
# pip install langchain_huggingface 설치 필수.
from ModelLoad import get_llm
llm = get_llm()
#%%
# 강의에서는 predict(), predict_message()를 사용했지만, 최근부터는 invoke()로 통합.
## 입력 프롬프트와 함께 대답함. Model load시 pipeline 설정에서 return_full_text=False 로 지정하면 입력 프롬프트 x.
from IPython.display import display, Latex
prompt = "삼각 함수에 대해서 설명 해주세요."
answer = llm.invoke(prompt)
print(answer)
#%%
# 추가적으로 아래와 같이 .stream() 함수를 사용하면 GPT 처럼 순차적으로 출력할 수 있게 할 수 있음.
for token in llm.stream(prompt):
    print(token, end='', flush=True)
#%%
# 위 처럼 단순하게 string type의 데이터를 넣어서 출력을 얻을 수도 있지만,
# prompt templete을 이용하여 유동적으로 prompt를 변경하여 입력도 가능함.
## 해당 내용 또한 강의에 없음.
from langchain_core.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template(
    "나에게 {animal}과 관련된 {story} 이야기해줘."
)
prompt = prompt_template.format(animal = "개구리", story = "무서운")
answer = llm.invoke(prompt)
print(answer)
#%%
from langchain_core.prompts import ChatPromptTemplate
# 모델들 중, chat templete 형식을 지켜야하는 모델들도 존재하는데 그럴 시 아래와 같이 작성
## Qwen 모델 라인도 동일함.
chat_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."),
    ("user", "{input}")
])
prompt = chat_prompt_template.format(input = "삼각 함수에 대해서 설명 해주세요.")
answer = llm.invoke(prompt)
answer = answer.replace("\\", "\\\\")
display(Latex(answer))
#%%
# LangChain에서는 prompt message type도 지정하여 입력할 수 있는데, 종류는 아래와 같음.
## 1. HumanMessage: 사람으로부터 주어진 message.
### 설명: 사용자가 모델에게 보내는 메시지를 나타냄. 실제 인간 사용자의 입력을 표현.
## 2. AiMessage: AI/assistant로부터 주어진 message.
### 설명: 모델이 사용자에게 보내는 응답을 나타냄. AI의 답변이나 생성된 내용을 표현.
## 3. SystemMessage: system으로부터 주어진 message.
### 설명: 시스템의 지시나 설정을 전달. 모델에게 특정 역할을 부여하거나 대화의 전반적 지침 제공.
## 4. FunctionMessage: function call로부터 주어진 message.
### 설명: 함수 호출의 결과 및 특정 기능 출력을 전달. 모델이 외부 도구나 API와 상호작용할 때 사용됨.
# 아래처럼 구현하면 chatbot처럼 작동시킬 수 있음.
from langchain.schema import SystemMessage, HumanMessage, AIMessage

system_prompt = "당신은 좋은 퀄리티의 회사 이름을 잘 지어주는 역할을 맡고 있습니다."
prompt1 = "컬러풀 양말을 만드는 회사의 좋은 이름에는 어떤 것들이 있을까요?"
messages = [SystemMessage(content = system_prompt),
            HumanMessage(content = prompt1)]
AI_message = llm.invoke(messages)
print(f"{''.join(AI_message.split(':')[1:])[1:]} \n ===========================================================")

prompt2 = "그 중 제일 괜찮은 회사 이름 하나만 추천 해주세요."
messages.append(AIMessage(content = "".join(AI_message.split(":")[1:])[1:]))
messages.append(HumanMessage(content = prompt2))
AI_message = llm.invoke(messages)
print(AI_message)
#%%
# Chatbot 예시 (강의 x)
## 위 예시를 이용하여, 간단한 chatbot을 만들어 봄.
system_prompt = "당신은 사용자의 질문에 대답을 잘해주는 유용한 조수입니다."
messages = [SystemMessage(content = system_prompt)]
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
# LangChain에서는 출력된 output에서 사용자가 원하는 형식으로 parsing하는 기본 함수도 지원함.
# 자동 parsing 같은 특별한 기능은 없지만, 아래와 같은 장점들이 있다고 함.
## 1. 일관된 인터페이스 제공 - BaseOutputParser를 사용해 커스텀 parser를 구현하면, 
## LangChain의 다른 구성 요소와 일관된 방식으로 통합 가능.
## 2. 재사용성 향상: 한 번 작성한 parser를 다양한 chain이나 모델 출력에 재사용 가능.
## 3. 에러 처리 및 예외 관리: BaseOutParser를 기반으로 한 parser는 공통된 에러 처리 로직 구현가능.
from langchain.schema import BaseOutputParser
import re

# Comma를 찾아 출력 list 형태로 만들어주는 parser
class CommaSeparatedListOutputParser(BaseOutputParser):
    def parse(self, text: str):
        # Assistant:, Human:, System: 등 prompt 형식을 지움
        clean_text = re.sub(r'(Assistant|Human|System):', '', text)
        clean_text = clean_text.strip()
        return [item.strip() for item in clean_text.split(",") if item.strip()]
## Parsing example
print(CommaSeparatedListOutputParser().parse("hi, bye"))
# %%
# 이제 위에서 배웠던 지식들을 활용하여 Prompt Template + LLM + OutputParser를 시도.
from langchain_core.prompts import ChatPromptTemplate
import re

template = """당신은 쉼표로 구분된 목록을 생성하는 유용한 조수입니다. \
사용자가 카테고리를 전달하면 해당 카테고리에 속하는 5개의 객체를 쉼표로 구분된 목록으로 생성합니다. \
오직 쉼표로 구분된 목록만 반환하고 그 이상은 반환하지 마세요.
"""
human_prompt = "{text}"
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("user", human_prompt)
])
chain = chat_prompt | llm | CommaSeparatedListOutputParser()
results = chain.invoke({"text": "색깔"})
# Qwen 모델 특성이 그런건지 모르겠지만 한 번씩 예기치 못한 출력 (Assistant: , Human: 포함 등)이 잦게 나타나 
# Parser 클래스를 정밀하게 구현해야함.
print(results)

# Parser class를 제외한 chain의 경우.
chain = chat_prompt | llm
results = chain.invoke({"text": "색깔"})
print(results)
# %%
