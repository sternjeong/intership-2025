#%%
from ModelLoad import ModelLoader
llm = ModelLoader.get_llm()
emb = ModelLoader.get_embedding()
tok = ModelLoader.get_tokenizer()
#%%
# 강의에서는 OpenAI의 embedding 모델을 사용하기에 transformers로 따로 구현하여 진행.
## 여러 string에 대해 embedding을 진행하기 위해서는 아래와 같이 진행.
embeddings = emb.embed_documents(
    [
        "안녕!", 
        "빨간색 공",
        "파란색 공",
        "붉은색 공",
        "푸른색 공"
    ]
)
len(embeddings), len(embeddings[0])
#%%
## 하나의 string을 embedding 하려면 아래와 같이 진행.
embedded_query = emb.embed_query("안녕!")
len(embedded_query), embedded_query[:5]
#%%
# 위와 같이 매번 embedding을 진행하는 것은 비효율 적임. [이미 고정된 데이터이므로]
# embedding된 데이터를 저장해놓으면 좋을 것으로 보임.
## 그래서 나온 것이 vector store [해당 강의에서는 chromadb를 사용함]
## pip install chromadb 필수 ***
from langchain.docstore.document import Document
sample_texts = [
    "안녕!",
    "빨간색 공",
    "파란색 공",
    "붉은색 공",
    "푸른색 공"
]
# Langchain의 document 타입으로 바꿔 list에 저장하여 활용.
documents = []
for item in range(len(sample_texts)): 
    page = Document(page_content = sample_texts[item])
    documents.append(page)
documents
#%%
# Chroma vector store를 이용하여 documents 저장.
from langchain.vectorstores import Chroma
vector_db = Chroma.from_documents(documents, emb)
#%%
# similarity_search()에는 입력 데이터뿐만 아니라, k와 filter가 존재함.
## k: query와 가장 유사한 상위 k개의 문서를 반환.
## filter: chroma에 저장된 metadata에 기반한 필터링
### filter 예시: category가 "sports"인 문서들일 경우 -> filter={"category":"sports"}
# similarity_search()는 입력으로 string을 사용하는 것
query = "빨간 공"
docs = vector_db.similarity_search(query)
print(docs[0].page_content)
print(docs[1].page_content)
print(docs[2].page_content)
print(docs[3].page_content)
#%%
# string 뿐 아니라, embedding도 입력으로 하여 유사도 검색이 가능.
query = "빨간 공"
embedding_vector = emb.embed_query(query)
docs = vector_db.similarity_search_by_vector(embedding_vector)
print(docs[0].page_content)
print(docs[1].page_content)
print(docs[2].page_content)
print(docs[3].page_content)
#%%
