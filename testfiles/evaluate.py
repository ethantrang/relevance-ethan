import pandas as pd
import sys

sys.path.append("./")

from models.llm_client import llm_client

# Load the CSV file
df = pd.read_csv('./evaluation/data/testset.csv')

# Define a function to generate answers using llm_client
def generate_outputs(row):
    question = row['question']
    result = llm_client.generate_with_openai(question, "nrma-car-pds-1023-east")
    answer = result["response"]
    texts = result["texts"]
    return pd.Series([answer, texts])

# Apply the function to the DataFrame and create a new 'answer' column
df[['answer', 'texts']] = df.apply(generate_outputs, axis=1)

# Save the updated DataFrame back to a CSV file
df.to_csv('./evaluation/data/results_1.csv', index=False)
