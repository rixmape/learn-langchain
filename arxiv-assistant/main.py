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
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import PromptTemplate, MessagesPlaceholder
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

from templates import answer_template, expansion_template


LLM_MODEL = "gpt-3.5-turbo-16k"


class AnswerQuestionInput(BaseModel):
    question: str = Field(description="The question to answer")
    abstracts: list[str] = Field(description="The abstracts to use")


def search(query: str) -> list[str]:
    search = arxiv.Search(query, max_results=4)
    # TODO: Consider the token limit of the LLM. Concatenate the abstracts
    # until the token limit is reached.
    return [result.summary for result in search.results()]


def expand_query(query: str) -> str:
    expansion_prompt_template = PromptTemplate(
        input_variables=["query"],
        template=expansion_template,
    )

    expansion_chain = LLMChain(
        llm=ChatOpenAI(temperature=0, model=LLM_MODEL),
        prompt=expansion_prompt_template,
        output_key="expanded_query",
    )

    return expansion_chain.predict(query=query)


def answer(abstracts: list[str], expanded_query: str) -> str:
    answer_prompt_template = PromptTemplate(
        input_variables=["abstracts", "expanded_query"],
        template=answer_template,
    )

    answer_chain = LLMChain(
        llm=ChatOpenAI(temperature=0, model=LLM_MODEL),
        prompt=answer_prompt_template,
        output_key="answer",
    )

    return answer_chain.predict(
        abstracts=abstracts,
        expanded_query=expanded_query,
    )


st.set_page_config(page_title="arXiv Assistant", page_icon="📖")
st.title("📖 arXiv Assistant")

# Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
# chat_history = MessagesPlaceholder(variable_name="chat_history")
memory = ConversationBufferMemory(
    chat_memory=msgs,
    # memory_key="chat_history",
    return_messages=True,
)

# Add a default message if there are no messages in memory
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

# Render current messages from StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

tools = [
    Tool(
        name="Expand Search Query",
        func=expand_query,
        description="useful for expanding a search query",
    ),
    Tool(
        name="Search Abstracts",
        func=search,
        description="useful for searching research abstracts given a query",
    ),
    StructuredTool.from_function(
        name="Answer Question",
        func=answer,
        description="useful for answering a question given some abstracts",
        args_schema=AnswerQuestionInput,
    ),
]

# If user inputs a new prompt, generate and draw a new response
# New messages are added to memory automatically
if query := st.chat_input():
    st.chat_message("user").write(query)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        llm = ChatOpenAI(temperature=0, model=LLM_MODEL)
        executor = initialize_agent(
            llm=llm,
            tools=tools,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            # memory=memory,
            # agent_kwargs={
            #     "memory_prompts": [chat_history],
            #     "input_variables": [
            #         "input",
            #         "agent_scratchpad",
            #         "chat_history",
            #     ],
            # },
        )

        response = executor.run(query, callbacks=[st_callback])
        st.write(response)
