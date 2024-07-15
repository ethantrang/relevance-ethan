from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers.pydantic import PydanticOutputParser # needed due to import conflict

from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from ragas import evaluate

import os
import ast
from datasets import Dataset 
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
import sys
sys.path.append("./")
from models.llm_client import llm_client
import os

class EvaluationClient: 

    def __init__(self):
        pass

    def create_testset(
            self, 
            data_path="loader/data", 
            generator_llm_model="gpt-3.5-turbo-16k", 
            critic_llm_model="gpt-4", 
            test_size=5,
            n_simple=0.5, 
            n_reasoning=0.25, 
            n_multi_context=0.25,
        ) -> bytes:

        loader = DirectoryLoader(data_path) # directory
        documents = loader.load()

        for document in documents:
            document.metadata['filename'] = document.metadata['source']

        generator_llm = ChatOpenAI(model=generator_llm_model)
        critic_llm = ChatOpenAI(model=critic_llm_model)
        embeddings = OpenAIEmbeddings()

        generator = TestsetGenerator.from_langchain(
            generator_llm,
            critic_llm,
            embeddings
        )

        testset = generator.generate_with_langchain_docs(
            documents, 
            test_size=test_size, 
            distributions={simple: n_simple, reasoning: n_reasoning, multi_context: n_multi_context},
            raise_exceptions=False
        )

        testset_bytes = testset.to_pandas().to_csv(index=False).encode('utf-8')

        return testset_bytes

    def generate_answers_and_texts(
            self,
            testset_path='evaluation/tests/testset.csv', 
            item_id="nrma-car-pds-1023-east",
            full_testset_path='evaluation/tests/full_testset_{id}.csv',
            testset_id=1,
            retrieval_method="vector", 
            generation_method="openai", 
        ) -> str:
        
        df = pd.read_csv(testset_path)

        def generate_outputs(row):
            question = row['question']
            result = llm_client.generate(question, item_id, retrieval_method, generation_method)
            answer = result["response"]
            texts = result["texts"]

            return pd.Series([answer, texts])

        df[['answer', 'texts']] = df.apply(generate_outputs, axis=1)

        df.to_csv(full_testset_path.format(id=testset_id), index=False)

        return full_testset_path.format(id=testset_id)
        
    def generate_evaluation(
            self,
            testset_id=1,
            full_testset_path='evaluation/tests/full_testset_{id}.csv',
        ) -> str:

        df = pd.read_csv(full_testset_path.format(id=testset_id))

        # df_sampled = df.sample(n=3, random_state=42)

        data_samples = {
            'question': df['question'].tolist(),
            'answer': df['answer'].tolist(),
            'contexts': df['texts'].apply(lambda x: ast.literal_eval(x)).tolist(),
            'ground_truth': df['ground_truth'].tolist()
        }

        dataset = Dataset.from_dict(data_samples)

        result = evaluate(
            dataset,
            metrics=[
                context_precision,
                faithfulness,
                answer_relevancy,
                context_recall,
            ],
            raise_exceptions=False
        )

        return result
    
    def load_testset(self, data: bytes, file_path: str):
        df = pd.read_csv(data)
        df.to_csv(file_path, index=False)
        return file_path
    
    def evaluate_one(
            self,
            data: bytes,
            retrieval_method: str, 
            generation_method: str,
            item_id: str,
            # testset_path='evaluation/tests/testset.csv', 
        ):
        
        # from data create testset_path 
        testset_path = self.load_testset(data, 'evaluation/tests/testset.csv')

        full_testset_path = self.generate_answers_and_texts(
            item_id=item_id,
            testset_path=testset_path,
            retrieval_method=retrieval_method,
            generation_method=generation_method,
        )
        result = self.generate_evaluation()

        os.remove(testset_path)
        os.remove(full_testset_path)

        return result
    
    def evaluate_all(
            self,
            data: bytes,
            item_id: str,
            # testset_path='evaluation/tests/testset.csv', 
        ):

        testset_path = self.load_testset(data, 'evaluation/tests/testset.csv')

        test_order = [
            {"retrieval_method": "vector", "generation_method": "openai"},
            {"retrieval_method": "vector", "generation_method": "anthropic"},
            {"retrieval_method": "hybrid", "generation_method": "openai"},
            {"retrieval_method": "hybrid", "generation_method": "anthropic"},
        ]
        results = []
        
        #! can't async because of rate limits
        for index, test in enumerate(test_order):

            full_testset_path = self.generate_answers_and_texts(
                item_id=item_id,
                testset_path=testset_path, 
                retrieval_method=test["retrieval_method"], 
                generation_method=test["generation_method"],
            )
            result = self.generate_evaluation(testset_id=index)
            results.append({
                "test_n": index,
                "retrieval_method": test["retrieval_method"],
                "generation_method": test["generation_method"],
                "result": result
            })

            os.remove(full_testset_path)

        # TODO: async job

        os.remove(testset_path)

        return results


evaluation_client = EvaluationClient()

if __name__=="__main__": 
    
    # evaluation_client.create_testset()
    # evaluation_client.generate_answers_and_texts()
    # result = evaluation_client.generate_evaluation()
    results = evaluation_client.evaluate_all()
    print(results)