import os.path
import json
from langchain.document_loaders import PyPDFLoader, JSONLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from data import jsontotext
class Index():
    def __init__(self,file):
        self.file = file
        if os.path.isfile(self.file):
            self.file = self.text_loader()
        self.text_split = CharacterTextSplitter(
            separator="\n",
            chunk_size= 500,
            chunk_overlap= 50
        )
        self.text = self.text_split.create_documents([self.file])
    def text_loader(self):
        if self.file.endswith('.pdf'):
            loader = PyPDFLoader(self.file)
            pages = loader.load_and_split()
            page_content = []
            for i,page in enumerate(pages):
                page_content.append(page)
            f = " ".join(page_content)
        elif self.file.endswith('.txt'):
            with open(self.file,'r',encoding='utf-8') as file:
                f = file.read()
        elif self.file.endswith('.json'):
            f = open(self.file)
            f = json.load(f)
            f = " ".join(jsontotext(f))
        return f
    def split_text(self):
        if os.path.isfile(self.file):
            self.file = self.text_loader()

        text_split = CharacterTextSplitter(
            separator="\n",
            chunk_size = 300,
            chunk_overlap = 50
        )
        text = text_split.create_documents([self.file])
        return text

    def load_build_index(self,embed_model):
        vectorsotores = FAISS.from_documents(self.split_text(), embed_model)
        return vectorsotores

    def retrieval(self,query,embed_model):
        vectorstores = FAISS.from_documents(self.text, embed_model)
        docs = vectorstores.similarity_search_with_score(query, k=5)
        docs = [f"{doc[0].page_content}\n" for doc in docs]
        docs = " ".join(docs)
        template = "Sử dụng các đoạn ngữ cảnh sau để trả lời câu hỏi ở cuối bằng tiếng Việt. Nếu không biết câu trả lời, bạn chỉ cần nói rằng bạn không biết, đừng cố bịa ra câu trả lời."
        prompt = f"{template} \n {docs} Câu hỏi: {query} \n Trả lời: "
        return prompt
