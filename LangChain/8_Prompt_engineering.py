#%%
# Prompt engineering 중, few-shot 생성에 관해 알아보자.
# Few-shot templete 모듈을 사용하여 손쉽게 생성 가능하다고 함.
from ModelLoad import ModelLoader
llm = ModelLoader.get_llm()
emb = ModelLoader.get_embedding()
tok = ModelLoader.get_tokenizer()
#%%
examples = [
  {
    "question": "Muhammad Ali와 Alan Turing 중 누가 더 오래 살았나요?",
    "answer":
"""
추가 질문이 필요한가요: 예.
추가 질문: Muhammad Ali가 사망했을 때 몇 살이었나요?
중간 답변: Muhammad Ali는 사망했을 때 74세였습니다.
추가 질문: Alan Turing이 사망했을 때 몇 살이었나요?
중간 답변: Alan Turing은 사망했을 때 41세였습니다.
그래서 최종 답변은: Muhammad Ali입니다.
"""
  },
  {
    "question": "Craigslist의 창립자는 언제 태어났나요?",
    "answer":
"""
추가 질문이 필요한가요: 예.
추가 질문: Craigslist의 창립자는 누구였나요?
중간 답변: Craigslist는 Craig Newmark에 의해 창립되었습니다.
추가 질문: Craig Newmark는 언제 태어났나요?
중간 답변: Craig Newmark는 1952년 12월 6일에 태어났습니다.
그래서 최종 답변은: 1952년 12월 6일입니다.
"""
  },
  {
    "question": "조지 워싱턴의 외할아버지는 누구였나요?",
    "answer":
"""
추가 질문이 필요한가요: 예.
추가 질문: 조지 워싱턴의 어머니는 누구였나요?
중간 답변: 조지 워싱턴의 어머니는 메리 볼 워싱턴(Mary Ball Washington)이었습니다.
추가 질문: 메리 볼 워싱턴의 아버지는 누구였나요?
중간 답변: 메리 볼 워싱턴의 아버지는 조셉 볼(Joseph Ball)이었습니다.
그래서 최종 답변은: 조셉 볼입니다.
"""
  },
  {
    "question": "'죠스'와 '카지노 로얄'의 감독이 같은 나라 출신인가요?",
    "answer":
"""
추가 질문이 필요한가요: 예.
추가 질문: '죠스'의 감독은 누구인가요?
중간 답변: '죠스'의 감독은 스티븐 스필버그(Steven Spielberg)입니다.
추가 질문: 스티븐 스필버그는 어느 나라 출신인가요?
중간 답변: 미국입니다.
추가 질문: '카지노 로얄'의 감독은 누구인가요?
중간 답변: '카지노 로얄'의 감독은 마틴 캠벨(Martin Campbell)입니다.
추가 질문: 마틴 캠벨은 어느 나라 출신인가요?
중간 답변: 뉴질랜드입니다.
그래서 최종 답변은: 아니요
"""
  }
]

#%%
# 아래는 romptTemplete을 이용하여 기본 Prompt를 생성하는 코드.
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate

example_prompt = PromptTemplate(input_variables=["question", "answer"], template="Question: {question}\n{answer}")

print(example_prompt.format(**examples[0]))
#%%
# CoT 형태로 프롬프트 만드는 코드.
prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Question: {input}",
    input_variables=["input"]
)

print(prompt.format(input="메리 볼 워싱턴의 아버지는 누구였나요?"))
#%%
# 유사한 few-shot 정보를 찾기 위한 vector store 정의.
## SemanticSimilarityExample 모듈을 이용해 가장 유사한 example 만을 우선 search.
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma
example_selector = SemanticSimilarityExampleSelector.from_examples(
    # 선택한 example들 설정
    examples,
    # 유사한 임베딩값을 찾기 위한 임베딩 모델 설정
    emb,
    # vector store 설정
    Chroma,
    # 몇개의 가장 유사한 embedding을 찾을 것인지를 설정
    k=1
)

# example들 중 가장 유사한 example을 찾기
question = "메리 볼 워싱턴의 아버지는 누구였나요?"
selected_examples = example_selector.select_examples({"question": question})
print(f"입력과 가장 유사한 예시: {question}")
for example in selected_examples:
    print("\n")
    for k, v in example.items():
        print(f"{k}: {v}")
#%%
# example_selector를 이용하여 가장 유사한 예제 1개를 통한 1-Shot prompt 구성.
prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    suffix="Question: {input}",
    input_variables=["input"]
)

print(prompt.format(input="메리 볼 워싱턴의 아버지는 누구였나요?"))
#%%
