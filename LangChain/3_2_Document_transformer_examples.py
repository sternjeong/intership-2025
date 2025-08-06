#%%
# 앞서 배운 DocumentsLoader를 이용하여 가지고 온 text의 경우 너무 내용이 방대함.
# 그래서 해당 섹션에서 긴 text 데이터를 좀 더 작은 text로 분할하는 transformation 과정을 배움.

from ModelLoad import ModelLoader
llm = ModelLoader.get_llm()
emb = ModelLoader.get_embedding()
tok = ModelLoader.get_tokenizer()
#%%
# 예제 txt 파일 다운로드 받기
# wget https://gist.githubusercontent.com/solaris33/ba8737e11f886b188de319884cffae47/raw/58f9861e7eefe9de75bd62be04f7b6082ec568d2/llm_example_text.txt -O llm_example_text.txt

with open("./llm_example_text.txt") as f: 
    llm_example_text = f.read()
llm_example_text
# 하나의 통 텍스트로 된 string을 부분부분 나누는 과정이 이번 과정의 핵심.
#%%
# 다양한 모듈 중, 가장 많이 사용되며, default로 사용되는 RevursiveCharacterTextSplitter 사용.
## 기본적으로 사용하는 구분 기호는 ["\n\n", "\n", " ", ""] 임.
## 파라미터들
### 1. length_function: chunk의 길이를 계산하는 방법을 지정. 
### 기본값으로는 문자의 길이로 하지만, token 개수로 지정하는 경우도 흔함.
### 2. chunk_size: (length_function에서 정의한 기준에 따른) chunk의 최대 크기.
### 3. chunk_overlap: chunk 간에 최대 중복 크기, 자연스러운 연결을 위해서 적절한 길이의 중복이 있는 것이 좋음.
### 4. add_start_index: 원본 문서내에서 chunk의 시작 위치를 metadata에 포함할 지 말지를 결정.
from langchain.text_splitter import RecursiveCharacterTextSplitter


def token_length(text):
    return len(tok.encode(text))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100, 
    chunk_overlap = 20,
    length_function = len, # 정말 다양한 형식으로 custom하여 응용이 가능하다고 함. ***** token개수는 token_length 함수로 변경.
    add_start_index = True
)

texts = text_splitter.create_documents([llm_example_text])
print(texts[0])
#%%
# Text를 통째로 사용하는 경우가 아닌 특수한 경우에서의 splitter
## HTMLHeaderTextSplitter
from langchain.text_splitter import HTMLHeaderTextSplitter

html_string = """
<!DOCTYPE html>
<html>
<body>
    <div>
        <h1>Foo</h1>
        <p>Some intro text about Foo.</p>
        <div>
            <h2>Bar main section</h2>
            <p>Some intro text about Bar.</p>
            <h3>Bar subsection 1</h3>
            <p>Some text about the first subtopic of Bar.</p>
            <h3>Bar subsection 2</h3>
            <p>Some text about the second subtopic of Bar.</p>
        </div>
        <div>
            <h2>Baz</h2>
            <p>Some text about Baz</p>
        </div>
        <br>
        <p>Some concluding text about Foo</p>
    </div>
</body>
</html>
"""

headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
]

# 지정한 headers 사이에 존재하는 데이터들을 parsing.
html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
html_header_splits = html_splitter.split_text(html_string)
html_header_splits
#%%
# 아래는 HTML splitter를 이용하여 split 후, Recursive splitter를 이용해서 세분화하는 과정.

url = "https://ko.wikipedia.org/wiki/%EB%8C%80%ED%98%95_%EC%96%B8%EC%96%B4_%EB%AA%A8%EB%8D%B8"

headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
    ("h4", "Header 4"),
]

html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

# 우선 URL을 입력하여 header 기준, HTML split 진행.
html_header_splits = html_splitter.split_text_from_url(url)
html_header_splits
#%%
# 아래와 같이 Recursive Split을 한 번 더 진행함.
chunk_size = 500
chunk_overlap = 30
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = chunk_size, chunk_overlap = chunk_overlap
)

splits = text_splitter.split_documents(html_header_splits)
splits[8:11]
#%%
# Programming language를 구분할 때
# CodeTextSplitter를 사용.
from langchain.text_splitter import(
    RecursiveCharacterTextSplitter,
    Language
)
# Language 모듈에서 지원하는 언어들 확인.
[e.value for e in Language]
#%%
# Python code를 split할 때 사용되는 인자 목록 확인.
RecursiveCharacterTextSplitter.get_separators_for_language(Language.PYTHON)

#%%
# Python code split 예시
PYTHON_CODE = """
def hello_world():
    print("Hello, World!")

# Call the function
hello_world()
"""
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=50, chunk_overlap=0
)
python_docs = python_splitter.create_documents([PYTHON_CODE])
python_docs

#%%
# Jave script 파일에 대한 split도 가능함.
RecursiveCharacterTextSplitter.get_separators_for_language(Language.JAVA)
#%%
JS_CODE = """
function helloWorld() {
  console.log("Hello, World!");
}

// Call the function
helloWorld();
"""

js_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.JS, chunk_size=60, chunk_overlap=0
)
js_docs = js_splitter.create_documents([JS_CODE])
js_docs
#%%
# Markdown도 가능함.
from langchain.text_splitter import MarkdownHeaderTextSplitter
markdown_document = "# Foo\n\n    ## Bar\n\nHi this is Jim\n\nHi this is Joe\n\n ### Boo \n\n Hi this is Lance \n\n ## Baz\n\n Hi this is Molly"

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_header_splits = markdown_splitter.split_text(markdown_document)
md_header_splits
#%%
markdown_document = "# Intro \n\n    ## History \n\n Markdown[9] is a lightweight markup language for creating formatted text using a plain-text editor. John Gruber created Markdown in 2004 as a markup language that is appealing to human readers in its source code form.[9] \n\n Markdown is widely used in blogging, instant messaging, online forums, collaborative software, documentation pages, and readme files. \n\n ## Rise and divergence \n\n As Markdown popularity grew rapidly, many Markdown implementations appeared, driven mostly by the need for \n\n additional features such as tables, footnotes, definition lists,[note 1] and Markdown inside HTML blocks. \n\n #### Standardization \n\n From 2012, a group of people, including Jeff Atwood and John MacFarlane, launched what Atwood characterised as a standardisation effort. \n\n ## Implementations \n\n Implementations of Markdown are available for over a dozen programming languages."

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
]

# MD splits
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_header_splits = markdown_splitter.split_text(markdown_document)
md_header_splits

# 위의 Python, Java scripts, Markdown 셋 다 recursive splitter로 추가 split 가능.
#%%
# 위에서는 계속 string 단위로 진행했다면, 이제는 token 단위로 split 하는 방법에 대해 기술.
# OpenAI에서 개발된 BPE(Byte-Pair-Encoding) tokenizer 사용 pip install tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
with open('./llm_example_text.txt') as f:
    llm_example_text = f.read()

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=0
)
texts = text_splitter.split_text(llm_example_text)
texts