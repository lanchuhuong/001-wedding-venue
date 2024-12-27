import os
from collections.abc import Iterator

import streamlit as st
from langchain.schema import Document
from langchain_core.messages import BaseMessageChunk
from langchain_openai import ChatOpenAI


def get_llm_response(
    query: str, context: list[Document], chat_history
) -> Iterator[str]:
    """Generate a response using GPT-4 based on the retrieved context"""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=st.session_state.OPENAI_API_KEY,
        seed=123456,
    )

    # Prepare the context string
    context_text = ""
    image_descriptions = ""

    for document in context:
        if document.metadata["type"] == "text":
            context_text += f"==========\nCompany: {document.metadata['company']}\n\n"
            context_text += f"Document Content: {document.page_content}\n\n==========\n"
        elif document.metadata["type"] == "image":
            image_descriptions += (
                f"==========\nCompany: {document.metadata['company']}\n\n"
            )
            image_descriptions += (
                f"Image Description: {document.page_content}\n\n==========\n"
            )

    # Create the prompt
    system_prompt = """
        You are a helpful assistant for a wedding venue search system. 
        The context provided to you contains information about wedding venues. 
        Use those contexts to answer user's questions about wedding venues.
        It is almost the case that there is always information in the context provided, never say the context does not provide information. 
        If the specific information isn't in the context at all, say so. 
        Be concise but informative."""

    messages = [
        {"role": "system", "content": system_prompt},
    ]

    if chat_history:
        messages.extend(chat_history)

    messages.append(
        {
            "role": "user",
            "content": f"Context: {context_text}\nImage Descriptions: {image_descriptions}\n\nQuestion: {query}",
        }
    )

    for chunk in llm.stream(messages):
        content = chunk.content.replace("$", r"\$") if chunk.content else ""
        yield content
