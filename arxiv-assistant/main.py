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

expansion_template = """You are a language model trained to perform query expansion. Given a query, you are expected to add synonyms. Here are some examples:

Query: What is Shor's algorithm?
Expanded query: Shor's algorithm OR factorization OR Peter Shor OR quantum computing OR quantum algorithm

Query: What is the Higgs boson?
Expanded query: Higgs boson OR Higgs particle OR Standard Model OR particle physics OR CERN OR Large Hadron Collider OR elementary particle

Query: Riemann hypothesis
Expanded query: Riemann hypothesis OR Riemann zeta function OR prime number theorem OR analytic continuation OR complex analysis OR Bernhard Riemann OR number theory

Query: What is the P versus NP problem?
Expanded query: P versus NP problem OR computational complexity theory OR polynomial time OR NP-hard OR NP-complete OR NP-intermediate OR Boolean satisfiability problem

Query: Navier-Stokes equation
Expanded query: Navier-Stokes equation OR fluid dynamics OR partial differential equation OR fluid mechanics OR turbulence OR incompressible flow OR viscous flow

Query: {query}
Expanded query:"""
expansion_prompt_template = PromptTemplate(
    input_variables=["query"],
    template=expansion_template,
)

answer_template = """You are a research professor at Harvard University. Please use these abstracts to provide a response to the user's query. If the information is not in the abstracts, you may use your own knowledge but you need to explicitly state that you are doing so.

Abstracts:

{abstracts}

Query: {expanded_query}
Answer:"""
answer_prompt_template = PromptTemplate(
    input_variables=["abstracts", "expanded_query"],
    template=answer_template,
)

# If user inputs a new prompt, generate and draw a new response
# New messages are added to memory automatically
if query := st.chat_input():
    st.chat_message("user").write(query)

    with st.chat_message("assistant"):
        sthandler = StreamHandler(st.empty())

        # TODO: Using an agent can be more efficient than using a chain.
        expansion_chain = LLMChain(
            llm=OpenAI(temperature=0),
            prompt=expansion_prompt_template,
            output_key="expanded_query",
        )

        expanded_query = expansion_chain.predict(query=query)

        papers = arxiv.Search(expanded_query, max_results=4)

        # TODO: Consider the token limit of the LLM. Concatenate the abstracts until the token limit is reached.
        abstracts = "\n\n".join([paper.summary for paper in papers.results()])

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
