from datasets import load_dataset
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from ragas import evaluate
import pandas as pd

from datasets import Dataset 
import ast

# Load the CSV file
df = pd.read_csv('evaluation/data/results_1.csv')

# df_sampled = df.sample(n=3, random_state=42)

# Transform the DataFrame into the desired format
data_samples = {
    'question': df['question'].tolist(),
    'answer': df['answer'].tolist(),
    'contexts': df['texts'].apply(lambda x: ast.literal_eval(x)).tolist(),
    'ground_truth': df['ground_truth'].tolist()
}

# Create a Dataset from the dictionary
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

print(result)

# {'context_precision': 0.8028, 'faithfulness': 0.9167, 'answer_relevancy': 0.9822, 'context_recall': 0.8333}
