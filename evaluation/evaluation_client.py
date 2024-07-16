import io
import os
import sys
from uuid import uuid4
import ast
from datasets import Dataset
import pandas as pd
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from ragas.metrics import answer_relevancy, faithfulness, context_recall, context_precision
from ragas import evaluate

load_dotenv()

import sys
sys.path.append("./")
from models.llm_client import llm_client
from loader.loader_client import loader_client

class EvaluationClient:

    def __init__(self):
        pass

    def create_testset(
            self,
            data: bytes,
            generator_llm_model="gpt-3.5-turbo-16k",
            critic_llm_model="gpt-4",
            test_size=10,
            n_simple=0.5,
            n_reasoning=0.25,
            n_multi_context=0.25,
    ) -> bytes:
        try:
            data_path = f"evaluation/data/{uuid4().hex}.pdf"
            with open(data_path, 'wb') as file:
                file.write(data)

            loader = PyPDFLoader("/Users/ethantrang/Documents/relevance-ethan/loader/data/POL011BA.pdf") 
            documents = loader.load()

            for document in documents:
                document.metadata['filename'] = document.metadata['source']

            generator_llm = ChatOpenAI(model=generator_llm_model)
            critic_llm = ChatOpenAI(model=critic_llm_model)
            embeddings = OpenAIEmbeddings()

            generator = TestsetGenerator.from_langchain(
                generator_llm,
                critic_llm,
                embeddings,
            )

            testset = generator.generate_with_langchain_docs(
                raise_exceptions=True,
                documents=documents,
                test_size=test_size,
                distributions={simple: n_simple, reasoning: n_reasoning, multi_context: n_multi_context},
            )

            testset_bytes = testset.to_pandas().to_csv(index=False).encode('utf-8')

            os.remove(data_path)

            return testset_bytes

        except Exception as e:
            return {"message": str(e)}

    def _generate_answers_and_texts(
            self,
            testset_path='evaluation/tests/testset.csv',
            item_id="nrma-car-pds-1023-east",
            full_testset_path='evaluation/tests/full_testset_{id}.csv',
            testset_id=1,
            retrieval_method="vector",
            generation_method="openai",
    ) -> str:
        try:
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
        
        except Exception as e:
            return {"message": str(e)}

    def _generate_evaluation(
            self,
            testset_id=1,
            full_testset_path='evaluation/tests/full_testset_{id}.csv',
    ) -> str:
        try:
            df = pd.read_csv(full_testset_path.format(id=testset_id))

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

        except Exception as e:
            return {"message": str(e)}

    def load_testset(self, data: bytes, file_path: str):
        try:
            df = pd.read_csv(io.BytesIO(data))
            df.to_csv(file_path, index=False)
            return file_path

        except Exception as e:
            return {"message": str(e)}

    def evaluate_one(
            self,
            data: bytes,
            retrieval_method: str,
            generation_method: str,
            item_id: str,
    ):
        try:
            testset_path = self.load_testset(data, 'evaluation/tests/testset.csv')

            full_testset_path = self._generate_answers_and_texts(
                item_id=item_id,
                testset_path=testset_path,
                retrieval_method=retrieval_method,
                generation_method=generation_method,
            )
            result = self._generate_evaluation()

            os.remove(testset_path)
            os.remove(full_testset_path)

            return result

        except Exception as e:
            return {"message": str(e)}

    def evaluate_all(
            self,
            data: bytes,
            item_id: str,
    ):
        try:
            testset_path = self.load_testset(data, 'evaluation/tests/testset.csv')

            test_order = [
                {"retrieval_method": "vector", "generation_method": "openai"},
                {"retrieval_method": "vector", "generation_method": "anthropic"},
                {"retrieval_method": "hybrid", "generation_method": "openai"},
                {"retrieval_method": "hybrid", "generation_method": "anthropic"},
            ]
            results = []

            for index, test in enumerate(test_order):

                full_testset_path = self._generate_answers_and_texts(
                    item_id=item_id,
                    testset_path=testset_path,
                    retrieval_method=test["retrieval_method"],
                    generation_method=test["generation_method"],
                )
                result = self._generate_evaluation(testset_id=index)
                results.append({
                    "test_n": index,
                    "retrieval_method": test["retrieval_method"],
                    "generation_method": test["generation_method"],
                    "result": result
                })

                os.remove(full_testset_path)

            os.remove(testset_path)

            return results

        except Exception as e:
            return {"message": str(e)}

evaluation_client = EvaluationClient()

if __name__ == "__main__":
    data = open("/Users/ethantrang/Documents/relevance-ethan/evaluation/tests/testset_POL011BA.csv", "rb").read()
    results = evaluation_client.evaluate_one(data, "vector", "openai", "POL011BA")
    print(results)

