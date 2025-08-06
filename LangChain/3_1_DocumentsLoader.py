#%%
from ModelLoad import ModelLoader
llm = ModelLoader.get_llm()
emb = ModelLoader.get_embedding()
#%%
# 외부 텍스트 데이터를 연결하기 위해서는 외부 텍스트를 불러오는 과정이 필수.
# LangChain에서는 외부 텍스트 데이터를 손쉽게 불러오기 위한 Document loaders 모듈을 제공.
# Document Loader의 종류를 크게 분류하면 아래와 같다고 함.
## 1. WebBaseLoader
## 2. CSV
## 3. File Directory
## 4. HTML
## 5. JSON
## 6. Markdown
## 7. PDF

#############################################################
# WebBaseLoader
## WebBaseLoader는 URL로부터 HTML 페이지의 모든 텍스트를 load함. (HTML 소스를 전부 긁어와서 연결)
## pip install bs4 필수
from langchain.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://www.espn.com/")
# 위 URL의 HTML document를 반환.
data = loader.load()
print(data)
#%%
# 여러 URL을 같이 load하는법. (list 형태로 주면 됨)
loader = WebBaseLoader(["https://ko.wikipedia.org/wiki/%EB%8C%80%ED%98%95_%EC%96%B8%EC%96%B4_%EB%AA%A8%EB%8D%B8", "https://google.com"])
docs = loader.load()
print(docs)
#############################################################

#%%
#############################################################
# CSVLoader
## CSVLoader는 CSV파일로부터 모든 텍스트를 load함.
## 각 row 하나를 하나의 document로 load함.
# Test를 위한 예시를 아래 명령어를 통해 save.
# wget https://gist.githubusercontent.com/solaris33/06a3e796ca5aa5f2802fffeaa2f492c1/raw/46394476d39e22e1b4bda873b36a35967f6314f8/mlb_teams_2012.csv -O mlb_teams_2012.csv
from langchain.document_loaders.csv_loader import CSVLoader
import pandas as pd
example_csv = pd.read_csv("./mlb_teams_2012.csv")
print(example_csv)

loader = CSVLoader(file_path = "./mlb_teams_2012.csv")
# 각 row 정보를 하나의 documents로 load.
data = loader.load()
data
#%%
# 추가적인 parsing을 진행하고 싶을 경우.
loader = CSVLoader(file_path = "./mlb_teams_2012.csv", 
                   csv_args = {
                       'delimiter': ',', # Column 간 구분자. 보통 ',' 사용.
                       'quotechar': '"', # Column 값이 구분자나 줄 바꿈 문자같은 특수 문자를 포함할 때, 해당 필드를 감싸는 데 사용되는 문자. 기본 값은 '"'
                       'fieldnames': ['MLB Team', 'Payroll in millions', 'wins'] # 각 column에 해당하는 field 이름들의 리스트. (원하는 이름으로 rename 가능)
})
detail_data = loader.load()
detail_data
#%% 
# Source를 특정 column으로 지정하고 싶을 경우.
loader = CSVLoader(file_path = "./mlb_teams_2012.csv",
                   source_column = "Team")
data = loader.load()
data
#############################################################

#%%
# DirectoryLoader
## DirectoryLoader는 폴더 안에 존재하는 파일들을 필터링해서 읽어옴.

# .md파일 load
## 추가적으로 unstructured 모듈이 필요함으로 pip install unstructured 설치.
from langchain.document_loaders import DirectoryLoader

# .md 파일 parsing을 위해 pip install "unstructured[md]" 설치. ("" 있어야함.)
loader = DirectoryLoader('./sample_data', glob = "**/*.md")
docs = loader.load()
docs
#%%
# loader_cls를 CSVLoader로 지정해서 읽어오기.
loader = DirectoryLoader(path = './sample_data', glob = "**/california_housing_test.csv", loader_cls = CSVLoader)
docs = loader.load()
docs
#############################################################

#%%
# HTML
## UnstructuredHTMLLoader는 html 파일안에 내용을 읽어옴.
## 아래 명령어를 통해 예시 html 파일 설치
### !wget https://gist.githubusercontent.com/solaris33/f104c7f49ba5db24ea06fd6dfb4c51f9/raw/600a889c5c0eea1c5b82a51877ade446f4bcdf59/sample_html.html -O sample_html.html
from langchain.document_loaders import UnstructuredHTMLLoader

loader = UnstructuredHTMLLoader("./sample_html.html")
data = loader.load()
data
#%%
# 위와 같은 방법으로 진행하면 parsing이 어렵게 되어있다고 함.
# 그래서 BeautifulSoup을 이용하여 텍스트를 추출해서 불러옴.
## 불필요한 Tag 같은 것을 제거하고 순수 text 정보를 불러들이며, 추가로 title 정보도 metadata에 추가됨.
### 특수한 type에 대한 document_loaders가 엄청 많으니 왠만한 상황에 대한 loader는 다 존재할 것이라고 함.
from langchain.document_loaders import BSHTMLLoader

loader = BSHTMLLoader("./sample_html.html")
data = loader.load()
data
#############################################################
#%%
# JSONLoader
## JSONLoader는 json 파일 안에 내용을 읽어옴.
## jq 라는 parsing을 위한 programming language가 존재한다고 함. (json 파일 내 특정 내용을 filtering 가능)
# pip install jq 설치.
from langchain.document_loaders import JSONLoader
from pprint import pprint

loader = JSONLoader(
    file_path = "./sample_data/anscombe.json",
    jq_schema = '.',
    text_content = False # loaded json 데이터 내용을 문자열 형식으로 처리 할 것인지
)
data = loader.load()
pprint(data)
#%%
# JSON 데이터 형태에 따라 parsing이 가능함.
## 해당 예시 json 파일은 []로 묶여있기에 이를 jq_schema로 지정한다면
loader = JSONLoader(
    file_path = './sample_data/anscombe.json',
    jq_schema = '.[]',
    text_content = False
)
data = loader.load()
pprint(data)

#%%
# 특정 key로 가져오고 싶을 때
loader = JSONLoader(
    file_path = "./sample_data/anscombe.json",
    jq_schema = ".[].Series",
    text_content = False
)
data = loader.load()
pprint(data)
#############################################################
#%%
# Markdown
## UnstructuredMarkdownLoader는 md 파일안에 내용을 읽어옴.
from langchain.document_loaders import UnstructuredMarkdownLoader
markdown_path = "./sample_data/README.md"
loader = UnstructuredMarkdownLoader(markdown_path)
data = loader.load()
data
#%%
# Low 데이터를 보고 싶으면 아래와 같이 정의.
loader = UnstructuredMarkdownLoader(markdown_path, mode = 'elements')
data = loader.load()
data
#############################################################

#%%
#############################################################
# PDF
## PyPDFLoader는 pdf 파일안에 내용을 읽어옴.
## pip install pypdf 필수
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("./llama1.pdf")
# 다른 모듈에서는 load를 하는데 PyPDF의 경우, load_and_split()을 사용함.
## Metadata 정보에 각 정보가 어떤 페이지에서 가지고 와졌는지도 표기되어 있음.
pages = loader.load_and_split()
#%%
# .pdf 파일 뿐만 아니라 url을 통해서도 가지고 올 수 있음.
loader = PyPDFLoader("https://arxiv.org/pdf/2302.13971.pdf")
pages = loader.load_and_split()
pages[0]


# 이외에도 엄청나게 많은 Loader type이 존재함으로 확인할 필요가 있음.