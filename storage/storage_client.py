from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from langchain_community.vectorstores.mongodb_atlas import MongoDBAtlasVectorSearch
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_core.documents.base import Document

from openai import OpenAI

from typing import List
import os 
from dotenv import load_dotenv
load_dotenv()

"""
    https://www.mongodb.com/docs/atlas/atlas-vector-search/tutorials/reciprocal-rank-fusion/

    PyMongo documentation: https://www.mongodb.com/docs/languages/python/pymongo-driver/current/   
"""

# TODO: want to use claude rerank 

class StorageClient: 

    def __init__(self, uri=os.getenv("MONGODB_URI")): 

        self.uri = uri
        self.db_name = "relevance_db"
        self.collection_name = "relevance_collection"
        self.mongodb_client = MongoClient(self.uri) # , server_api=ServerApi('1')
        self.collection = self.mongodb_client[self.db_name][self.collection_name]
        self.embeddings = OpenAIEmbeddings()
        self.openai_client = OpenAI()

    def create_index(self):
        pass

    def delete_index(self):
        pass

    def store_documents(self, docs: List[Document]):
        docsearch = MongoDBAtlasVectorSearch.from_documents(
            documents=docs,
            embedding=self.embeddings,
            collection=self.collection, 
            index_name="relevance_index",
            relevance_score_fn="cosine"
        )
        return docsearch
    
    def generate_embedding(self, query: str) -> List[float]:

        return self.openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=query,
            encoding_format="float"
        ).data[0].embedding
    
    def retrieve_with_vector_search(self, query, item_id):
        results = self.collection.aggregate([
            {
                "$vectorSearch": {
                "index": "relevance_index",
                "path": "embedding",
                "queryVector": self.generate_embedding(query),
                "numCandidates": 10, # number of nearest neighbors 
                "limit": 4,
                }
            },
            {
                "$match": {
                    "item_id": f"{item_id}"  # Filtering condition
                }
            }
        ])

        texts = [doc['text'] for doc in results if 'text' in doc]

        return texts
    
    def retrieve_with_text_search(self, query, item_id):
        results = self.collection.aggregate([
            {
                "$search": {
                    "index": "relevance_index_text",
                    "text": {
                        "query": query,
                        "path": {
                            "wildcard": "*"
                        }
                    }
                },
            },
            {
                "$sort": {
                    "score": { "$meta": 'textScore' }
                },
            },
            {   
                "$match": {
                    "item_id": f"{item_id}"  # Filtering condition
                }
            },
            {"$limit": 2}  # Limit the number of results to the top 5
        ])

        texts = [doc['text'] for doc in results if 'text' in doc]

        return texts
    
    def retrieve_with_hybrid_search(self, query, item_id): 
        
        # relying on vector search and enhanced with text search
        # inspired by https://www.mongodb.com/docs/atlas/atlas-vector-search/tutorials/reciprocal-rank-fusion/#about-the-query
        
        vector_texts = self.retrieve_with_vector_search(query, item_id)
        text_texts = self.retrieve_with_text_search(query, item_id)
        combined_texts = list(set(vector_texts + text_texts))

        return combined_texts

    def delete_documents(self, item_id):
        result = self.collection.delete_many({"item_id": item_id})
        return result.deleted_count  # Returns the count of deleted documents

storage_client = StorageClient()

if __name__=="__main__": 
    # import sys
    # sys.path.append("./")
    # from loader.loader_client import loader_client

    # data = open("./loader/data/nrma-car-pds-1023-east.pdf", "rb").read()
    # docs = loader_client.get_docs_from_pdf(data, "nrma-car-pds-1023-east")

    # response = storage_client.store_documents(docs)

    # print(response)

    query = "what is the product disclosure statement"

    # embedding = storage_client.generate_embedding(query)

    # print(embedding)

    results = storage_client.retrieve_with_hybrid_search(query, "nrma-car-pds-1023-east")
    # results = storage_client.delete_documents("nrma-car-pds-1023-east")

    print(results)