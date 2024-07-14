
SYSTEM_PROMPT = """
You are a helpful assistant.
"""

USER_PROMPT = """
```{context}```

Using the context above delimited by the triple backticks, answer the user's question. 

User: {query}
Assistant: 
"""