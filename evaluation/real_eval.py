from datasets import load_dataset
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from ragas import evaluate

dataset_qa = load_dataset("csv", data_files="evaluation/data.csv")

result = evaluate(
    dataset_qa["train"], # ["eval"]
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
    ],
)

df = result.to_pandas().to_csv("evaluation/results.csv")