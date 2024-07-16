# Relevance AI - Ethan's Take Home Assignment AI Engineer 

## Task approach:
- PDF data loaded and uploaded to MongoDB vector databases
- Developed two retrieval approaches: vector search and hybrid search (hybrid being a combination of vector and text search)
- Developed two generation frameworks: OpenAI gpt-3.5-turbo, Anthropic claude-3-sonnet 
- Evaluated pipelines using Ragas for the following metrics: retrieval (context precision, context recall), generation (faithfulness, answer relevancy)
- Production structure achieved by assuming a microservice-like/modularized project organization + REST API deployment

## Tech stack:
- Python: FastAPI, Uvicorn
- Frameworks: LangChain, LiteLLM, Ragas (for evaluation)
- AI Models: OpenAI GPT, Anthropic Claude (run from AWS Bedrock)
- Database: MongoDB Atlas

## Evaluation results: 
Disclaimer: Ragas I've found is slightly buggy in production, these are results when running the evaluations locally. 

The testsets used found in the evaluation/tests directory.

### On POL011BA.pdf (Allianz Personal Motor Insurance)

| Test Number | Retrieval Method | Generation Method | Context Precision | Faithfulness | Answer Relevancy | Context Recall |
|-------------|------------------|-------------------|-------------------|--------------|------------------|----------------|
| 0           | vector           | openai            | 1.0000            | 0.8942       | 0.9567           | 0.8000         |
| 1           | vector           | anthropic         | 1.0000            | 0.9000       | 0.9348           | 0.8000         |
| 2           | hybrid           | openai            | 1.0000            | 0.6571       | 0.9307           | 0.8750         |
| 3           | hybrid           | anthropic         | 1.0000            | 0.8833       | 0.9381           | 0.9000         |

### On nrma-car-pds-1023-east.pdf (NRMA Motor Insurance)

| Test Number | Retrieval Method | Generation Method | Context Precision | Faithfulness | Answer Relevancy | Context Recall |
|-------------|------------------|-------------------|-------------------|--------------|------------------|----------------|
| 0           | vector           | openai            | 1.0000            | 0.8455       | 0.9743           | 0.6667         |
| 1           | vector           | anthropic         | 1.0000            | 0.7933       | 0.9469           | 0.6667         |
| 2           | hybrid           | openai            | 1.0000            | 1.0000       | 0.9824           | 0.4167         |
| 3           | hybrid           | anthropic         | 1.0000            | 0.5429       | 0.9595           | 0.7667         |
