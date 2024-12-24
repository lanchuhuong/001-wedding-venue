import os
from collections.abc import Iterator

import streamlit as st
from langchain_core.messages import BaseMessageChunk
from langchain_openai import ChatOpenAI


def get_llm_response(
    query: str, context: dict, chat_history
) -> Iterator[BaseMessageChunk]:
    """Generate a response using GPT-4 based on the retrieved context"""
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=st.session_state.OPENAI_API_KEY,
        seed=123456,
    )

    # Prepare the context string
    context_text = ""
    image_descriptions = []

    for doc_id, content in context.items():
        context_text += f"\nCompany: {content['company']}\n"
        for text_doc in content["text"]:
            context_text += f"{text_doc.page_content}\n"
        for image_doc in content["images"]:
            image_descriptions.append(image_doc.page_content)

    # Create the prompt
    system_prompt = """
        You are a helpful assistant for a wedding venue search system. 
        The context provided to you contains information about wedding venues. 
        Use those contexts to answer user's questions about wedding venues.
        It is almost the case that there is always information in the context provided, never say the context does not provide information. 
        If the specific information isn't in the context at all, say so. 
        Be concise but informative."""

    # Start with system prompt
    messages = [
        {"role": "system", "content": system_prompt},
    ]

    # Add previous chat history if it exists
    if chat_history:
        messages.extend(chat_history)

    # Add current query with context as the latest message
    messages.append(
        {
            "role": "user",
            "content": f"Context: {context_text}\nImage Descriptions: {'; '.join(image_descriptions)}\n\nQuestion: {query}",
        }
    )

    response = llm.stream(messages)
    return response
