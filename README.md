# Reverse Ticket Search LLM

### What it is

This project is an example of a pre-MVP and some of the work that it took to get there. It has a combination of research/test files and the pre-MVP file (Pre_MVP_RAG.py).

### Why Ollama and LlamaIndex?

I initially wrote low level code that accomplished the same functionality as Ollama and LlamaIndex. However, the project had to be easily maintainable by future engineers (and perhaps non-engineers). To ensure easy (or easier) readability and maintainability, I decided to use Ollama and LlamaIndex.

### What does it do?

The purpose of this project is to find historical tickets with RAG using an on-premise process. It takes the database (currently a .csv file) and vectorizes it into the vectorstore utilizing LlamaIndex. The user or another program will submit a query describing a ticket's problem. Then the embedding model is able to vectorize the query, search the vectorstore, and return relevant tickets. Those tickets are then prompt engineered into a three shot prompt. Before the prompt is sent to the LLM, Ollama is started and the LLM is loaded into the system. Then the LLM outputs step by step instructions on how the previous tickets were solved (this is to mitigate bad documentation).


### How to use

I have not tested this to make sure it can be simply cloned and ran yet (without downloading special libraries and packages), as it is pre-production code. However, it is necessary to download Ollama and the LLM model you will be using from Ollama. Next, you want to run Pre_MVP_RAG.py to test the project. For your query, you can look through the .csv to see what tickets are in the test database to run against. Or you can say "My laptop is buzzing." to get a result.
