import streamlit as st

from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory

st.set_page_config(page_title="arXiv Assistant", page_icon="📖")
st.title("📖 arXiv Assistant")

# Set up memory
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    chat_memory=msgs,
    return_messages=True,
    memory_key="chat_history",
    output_key="output",
)

# Add a default message if there are no messages in memory
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

# Render current messages from StreamlitChatMessageHistory
view_messages = st.expander("View the message contents in session state")

# Set up the ComversationalAgent, passing in memory
agent_executor = initialize_agent(
    llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k"),
    tools=load_tools(["arxiv"]),
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    return_intermediate_steps=True,
    handle_parsing_errors=True,
    verbose=True,
)

# Render current messages from StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# If user inputs a new prompt, generate and draw a new response
if prompt := st.chat_input():
    st.chat_message("human").write(prompt)
    st_callback = StreamlitCallbackHandler(st.container())

    with st.chat_message("assistant"):
        response = agent_executor(prompt, callbacks=[st_callback])
        st.write(response["output"])

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
