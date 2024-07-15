# relevance-ethan

Relevance AI - Ethan's Take Home Assignment AI Engineer 

Task approach:
- PDF Data loaded and uploaded to MongoDB vector databases
- Developed two retrieval approaches: vector search and hybrid search (combination of vector and text index search)
- Developed two generation frameworks: OpenAI gpt-3.5-turbo, Anthropic claude-3-sonnet 
- Evaluated pipelines using Ragas for the following metrics: retrieval (context precision, context recall), generation (faithfulness, answer relevancy)
- Production structure achieved by assuming a microservice-like/modularized organization and REST API deployment

Tech stack:
- Python: FastAPI, Uvicorn
- Frameworks: LangChain, LiteLLM, Ragas (for evaluation)
- AI Models: OpenAI GPT, Anthropic Claude (run AWS Bedrock)
- Database: MongoDB

