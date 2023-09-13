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

import arxiv
import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import PromptTemplate

from templates import answer_template, expansion_template


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text + "▮")


st.set_page_config(page_title="arXiv Assistant", page_icon="📖")
st.title("📖 arXiv Assistant")

# Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
memory = ConversationBufferMemory(chat_memory=msgs)

# Add a default message if there are no messages in memory
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

# Render current messages from StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# If user inputs a new prompt, generate and draw a new response
# New messages are added to memory automatically
if query := st.chat_input():
    st.chat_message("user").write(query)

    with st.chat_message("assistant"):
        sthandler = StreamHandler(st.empty())

        # TODO: Using an agent can be more efficient than using a chain.
        expansion_prompt_template = PromptTemplate(
            input_variables=["query"],
            template=expansion_template,
        )

        expansion_chain = LLMChain(
            llm=OpenAI(temperature=0),
            prompt=expansion_prompt_template,
            output_key="expanded_query",
        )

        expanded_query = expansion_chain.predict(query=query)

        papers = arxiv.Search(expanded_query, max_results=4)

        # TODO: Consider the token limit of the LLM. Concatenate the abstracts until the token limit is reached.
        abstracts = "\n\n".join([paper.summary for paper in papers.results()])

        answer_prompt_template = PromptTemplate(
            input_variables=["abstracts", "expanded_query"],
            template=answer_template,
        )

        answer_chain = LLMChain(
            llm=OpenAI(temperature=0, streaming=True, callbacks=[sthandler]),
            prompt=answer_prompt_template,
            output_key="answer",
        )

        answer = answer_chain.predict(
            abstracts=abstracts,
            expanded_query=expanded_query,
        )

        sthandler.container.empty()
        st.write(answer)
