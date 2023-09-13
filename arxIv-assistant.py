"""
This module integrates LangChain, a Python framework for large language models (LLMs), and arXiv, a repository for e-prints of scientific papers, to create a question answering program. The goal is to demonstrate how to use LangChain and related libraries in building LLM-powered applications.

Psuedocode:

1. Ask the user for a query.
2. Expand the query by adding related words.
3. Use the expanded query to search arXiv for papers.
5. Create embeddings from the papers' abstracts.
6. Create embeddings from the orginal query.
6. Find the most relevant papers to the orginal query.
7. Feed the retrieved abstracts and the query to GPT-3.
8. Display the response.
9. Repeat steps 1-8 until the user quits.
"""
