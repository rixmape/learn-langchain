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

import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
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

# Set up expander to view the messages stored in memory
view_messages = st.expander("View the message contents in session state")

# Add a default message if there are no messages in memory
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

# Render current messages from StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

template = """You are a research professor at Harvard University. Your goal is to answer questions related to physics, mathematics, computer science, and other scientific disciplines. Always shorten your answers.

{history}
Student: {input}
Professor: """

prompt = PromptTemplate(
    input_variables=["history", "input"],
    template=template,
)

# If user inputs a new prompt, generate and draw a new response
# New messages are added to memory automatically
if query := st.chat_input():
    st.chat_message("user").write(query)

    with st.chat_message("assistant"):
        sthandler = StreamHandler(st.empty())
        llm = ChatOpenAI(temperature=0, streaming=True, callbacks=[sthandler])
        print(memory.load_memory_variables({}))
        llm_chain = ConversationChain(llm=llm, memory=memory, prompt=prompt)
        response = llm_chain.predict(input=query)
        sthandler.container.markdown(response)

# Draw the messages at the end, so newly generated ones show up immediately
with view_messages:
    """
    Memory initialized with:
    ```python
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    memory = ConversationBufferMemory(chat_memory=msgs)
    ```

    Contents of `st.session_state.langchain_messages`:
    """
    view_messages.json(st.session_state.langchain_messages)
