from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document


from typing import List
from uuid import uuid4
from datetime import datetime
import os

class LoaderClient:
    def __init__(self, storage_dir="./loader/data", chunk_size=2000, chunk_overlap=200, min_char_length=3000):

        self.chunk_size = chunk_size 
        self.chunk_overlap = chunk_overlap
        self.min_char_length = min_char_length

        self.storage_dir = storage_dir
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)
    
    def _generate_pdf_path(self):
        return os.path.join(self.storage_dir, f"{uuid4().hex}.pdf")
    
    def decode_bytes_to_pdf(self, data: bytes) -> str:
        output_path = self._generate_pdf_path()
        with open(output_path, 'wb') as file:
            file.write(data)
        return output_path
        
    def remove_pdf_from_dir(self, file_path: str) -> None:
        try:
            os.remove(file_path)
            print(f"File {file_path} removed successfully.")
        except FileNotFoundError:
            print(f"File {file_path} does not exist.")

    def resize_docs(self, docs: List[Document]) -> List[Document]: 

        new_docs = [] 
        curr_page_content = ""
        for doc in docs: 
            if len(curr_page_content) < self.min_char_length: 
                curr_page_content += doc.page_content
            else: 
                new_docs.append(Document(page_content=curr_page_content, metadata=doc.metadata))
                curr_page_content = doc.page_content
                
        return new_docs
    
    def get_docs_from_pdf(self, data: bytes, item_id: str) -> list:
        output_path = self.decode_bytes_to_pdf(data)
        loader = PyPDFLoader(output_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        docs = loader.load_and_split(text_splitter=text_splitter)

        for doc in docs:
            doc.metadata = {
                "item_id": item_id,
            }

        docs = self.resize_docs(docs)
        self.remove_pdf_from_dir(output_path)

        return docs

loader_client = LoaderClient()

if __name__ == '__main__':
    
    data = open("./loader/data/nrma-car-pds-1023-east.pdf", "rb").read()

    docs = loader_client.get_docs_from_pdf(data, "item_id")

    print(docs)