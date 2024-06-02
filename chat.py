from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import streamlit as st
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.schema.messages import HumanMessage,AIMessage

def main():
    prompt = ChatPromptTemplate(
        messages=[
SystemMessagePromptTemplate.from_template(
"""You are a helpful assistant.

> ``````"""),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{question}")])

    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    if len(msgs.messages) == 0:
        msgs.add_ai_message("Hello! How can I assist you today?")

    USER_AVATAR = "👤"
    BOT_AVATAR = "🤖"
    
    for msg in msgs.messages:
        avatar = USER_AVATAR if msg.type == "human" else BOT_AVATAR
        st.chat_message(msg.type,avatar=avatar).write(msg.content)
    
    if prompt := st.chat_input():
        st.chat_message("human",avatar=USER_AVATAR).write(prompt)

        with st.chat_message("assistant",avatar=BOT_AVATAR):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                from chain import chain
                _chat_history = st.session_state.langchain_messages[0:40]
                response = chain.stream({"question":prompt,"chat_history":_chat_history})

                for res in response:
                    full_response += res or "" 
                    message_placeholder.markdown(full_response + "|")
                    message_placeholder.markdown(full_response)

                msgs.add_user_message(prompt)
                msgs.add_ai_message(full_response)

            except Exception as e:
                st.error(f"An error occured. {e}")