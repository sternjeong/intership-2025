#%%
# 특정 input data를 구조화된 데이터로 추출하는 (entity extract) 과정을 진행해보자.
# Function call을 사용하여 진행하는 것으로 보임.
# Entity extraction 과정에서 추출한 데이터를 이용하여 다양한 부분에서 사용할 수 있음.
## 예로, 쇼핑몰에서 리뷰가 긍정인지 부정인지를 판단하여 부정적인 데이터만을 가지고 추가 학습을 시키거나 하는...
from ModelLoad import ModelLoader
llm = ModelLoader.get_llm()
emb = ModelLoader.get_embedding()
tok = ModelLoader.get_tokenizer()
#%%
# Entity extraction을 위해서는 create_extraction_chain을 우선 정의해야 함.
# 아래 코드는 실행이 되지 않을 것으로 보임. function_call()이 내장되어 있는 모델만 사용이 가능하다고 함.
from langchain.chains import create_extraction_chain

# 스키마(Schema) 설정
schema = {
    "properties": {
        "name": {"type": "string"},
        "height": {"type": "integer"},
        "hair_color": {"type": "string"},
    },
    "required": ["name", "height"],
}

# Input
inp = """알렉스는 키가 5피트입니다. 클라우디아는 알렉스보다 1피트 더 크고 그보다 더 높이 점프합니다. 클라우디아는 갈색 머리이고 알렉스는 금발입니다."""

# Run chain
chain = create_extraction_chain(schema, llm)
chain.invoke(inp)
#%%
# pip install langchain openai 필수 ***

## 아래는 OpenAI용 tool을 사용하기 위한 서버를 띄워놓기 위함 (ghye)
# export CUDA_VISIBLE_DEVICES=0,1
# vllm serve /DATA/XQbot/local_models/Qwen2.5-Coder-7B-Instruct \
#     --tokenizer Qwen/Qwen2.5-Coder-1.5B \
#     --enable-auto-tool-choice \
#     --tool-call-parser hermes \
#     --port 8001 \
#     --dtype auto

# 아래 코드는 function을 지정하여 강제로 해당 값으로 출력하게 만드는 function calling 예시 (ghye)
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def multiple(x: float, y: float) -> float:
    """Multiply two integers together."""

    return x * y

@tool
def add(x: int, y: int) -> int:
    """Add two integers together."""
    return x + y

tools = [add, multiple]

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8001/v1"
model_path = "/DATA/XQbot/local_models/Qwen2.5-Coder-7B-Instruct"

llm = ChatOpenAI(
        model=model_path,
        openai_api_key=openai_api_key, 
        openai_api_base=openai_api_base,
        max_tokens=512,
        temperature=0.7
        )
        
llm_with_tools = llm.bind_tools(tools)

query = "What is 3*12? Also, what is 11 + 49?"

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("You are Qwen, created by Alibaba Cloud. You are a helpful assistant."),
    HumanMessagePromptTemplate.from_template("{user_input}"),
])

from langchain_core.output_parsers import StrOutputParser
chain = chat_prompt | llm_with_tools | StrOutputParser()

print(chain.invoke({"user_input": query}))
# 추가로 1_Quickstart.py의 parsing 부분과 같이 data parsing 함수를 생성하여 사용할 수도 있음.
## 그러나 ChatOpenAI의 function calling 처럼 모델 레벨에서 출력을 강제하지는 않음.
### 정말 완벽하게 출력을 해야할 경우, function calling이 더 성능이 높을 수 밖에 없음.