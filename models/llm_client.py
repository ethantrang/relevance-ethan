import litellm
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

from typing import List

# local imports
from storage.storage_client import storage_client
from prompts import * 

class LLMClient: 

    def __init__(self):
        pass

    def create_context(self, query, item_id, retrieval_method="vector"):
        if retrieval_method == "vector":
            texts: List[str] = storage_client.retrieve_with_vector_search(query, item_id)
        elif retrieval_method == "hybrid":
            texts: List[str] = storage_client.retrieve_with_hybrid_search(query, item_id)
        return "\n".join(texts)

    def generate_with_openai(self, query, item_id):
        
        context = self.create_context(query, item_id)

        if not context:
            return "No relevant information found."

        response = litellm.completion(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=[
                {"role": "system", "content": f"{SYSTEM_PROMPT}"},
                {"role": "user", "content": f"{USER_PROMPT.format(context=context, query=query)}"}
            ]
        )

        return response.choices[0].message.content
    
    def generate_with_anthropic(self, query, item_id):

        context = self.create_context(query, item_id)

        if not context:
            return "No relevant information found."

        response = litellm.completion(
            model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
            temperature=0,
            messages=[
                {"role": "system", "content": f"{SYSTEM_PROMPT}"},
                {"role": "user", "content": f"{USER_PROMPT.format(context=context, query=query)}"}
            ]
        )

        return response.choices[0].message.content

llm_client = LLMClient()

if __name__=="__main__": 
    query = "What type of insurance do you have"
    item_id = "nrma-car-pds-1023-east"

    response = llm_client.generate_with_anthropic(query, item_id)

    print(response)