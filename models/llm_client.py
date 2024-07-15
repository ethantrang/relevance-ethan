import litellm
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

from typing import List

# local imports
from storage.storage_client import storage_client

from models.prompts import * 

class LLMClient: 

    def __init__(self):
        pass

    def generate(self, query, item_id, retrieval_method, generation_method):
        if generation_method == "openai":
            return self.generate_with_openai(query, item_id, retrieval_method)
        elif generation_method == "anthropic":
            return self.generate_with_anthropic(query, item_id, retrieval_method)

    def create_context(self, query, item_id, retrieval_method):
        if retrieval_method == "vector":
            texts: List[str] = storage_client.retrieve_with_vector_search(query, item_id)
        elif retrieval_method == "hybrid":
            texts: List[str] = storage_client.retrieve_with_hybrid_search(query, item_id)
        return "\n".join(texts), texts

    def generate_with_openai(self, query, item_id, retrieval_method):
        
        context_string, texts = self.create_context(query, item_id, retrieval_method)

        if not context_string:
            return "No relevant information found."

        response = litellm.completion(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=[
                {"role": "system", "content": f"{SYSTEM_PROMPT}"},
                {"role": "user", "content": f"{USER_PROMPT.format(context=context_string, query=query)}"}
            ]
        )

        return {
            "response": response.choices[0].message.content,
            "texts": texts
        }
    
    def generate_with_anthropic(self, query, item_id, retrieval_method):

        context_string, texts= self.create_context(query, item_id, retrieval_method)

        if not context_string:
            return "No relevant information found."

        response = litellm.completion(
            model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
            temperature=0,
            messages=[
                {"role": "system", "content": f"{SYSTEM_PROMPT}"},
                {"role": "user", "content": f"{USER_PROMPT.format(context=context_string, query=query)}"}
            ]
        )

        return {
            "response": response.choices[0].message.content,
            "texts": texts
        }

llm_client = LLMClient()

if __name__=="__main__": 
    query = "What type of insurance do you have"
    item_id = "nrma-car-pds-1023-east"

    response = llm_client.generate_with_anthropic(query, item_id)

    print(response)