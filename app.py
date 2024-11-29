# import openai
# import os
# import pandas as pd
# import streamlit as st
# import streamlit.components.v1 as components
# import yaml
# from azure.keyvault.secrets import SecretClient
# from streamlit_feedback import streamlit_feedback
# import extra_streamlit_components as stx
# import tiktoken


# # # loading CSS and HTML
# # def load_css(file_name="static/css/default.css"):
# #     with open(file_name) as f:
# #         st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# # def load_html(file_path):
# #     with open(file_path, "r", encoding="utf-8") as f:
# #         html_string = f.read()
# #     components.html(html_string, height=600)


# # def get_manager():
# #     # Get cookies
# #     return stx.CookieManager()


# # # Initialize CookieManager
# # cookie_manager = get_manager()


# # print(cookie_manager.get_all())


# # def count_token(text):
# #     return num_tokens_from_string(text, encoding_name="cl100k_base")


# # def num_tokens_from_string(string: str, encoding_name: str) -> int:
# #     """Returns the number of tokens in a text string."""
# #     encoding = tiktoken.get_encoding(encoding_name)
# #     num_tokens = len(encoding.encode(string))
# #     return num_tokens


# # st.title("Wedding Venue - ChatDocument")

# # # Initialize chat history
# # chat_history = []


# # # Initialize session state for variables if they don't exist
# # if "chat_history" not in st.session_state:
# #     st.session_state.chat_history = []

# st.title("💬 Chatbot")
# if "chat_history" not in st.session_state:
#     st.session_state["chat_history"] = [
#         {"role": "assistant", "content": "How can I help you?"}
#     ]

# for msg in st.session_state.chat_history:
#     st.chat_message(msg["role"]).write(msg["content"])

# if user_input := st.chat_input():
#     st.session_state.chat_history.append({"role": "user", "content": user_input})
#     st.chat_message("user").write(user_input)

#     with st.spinner(f"Generating response..."):
#         try:
#             if (
#                 "task_selected" in st.session_state
#                 and st.session_state["task_selected"]
#             ):
#                 response = get_completion_from_messages(
#                     st.session_state.chat_history,
#                     context,
#                     task=st.session_state["task_selected"],
#                 )
#             else:
#                 response = get_completion_from_messages(
#                     st.session_state.chat_history, context
#                 )

#         except openai.error.InvalidRequestError as e:
#             # Check if the error is due to token limit
#             if "model's maximum context length" in str(e):
#                 response = "Token limit reached. Please clear the conversation to continue. Please copy any needed information before clearing the session."  # Set a response for the exception case
#         except ValueError as e:
#             response = "Oops, I made a mistake. Please try again."
#         prompt_token_count = count_token(user_input)
#         st.chat_message("assistant").write(response)

#         # Count tokens for model's output
#         output_token_count = count_token(response)

#         # Append assistant's response to chat history
#         st.session_state.chat_history.append({"role": "assistant", "content": response})

# st.markdown(
#     """
# <div style='text-align: center; color: #0B5D8E; margin-bottom: 10px;'>
#     <h4>Your Opinion Matters!</h4>
#     <p>Please click a face to rate your experience.</p>
# </div>
# """,
#     unsafe_allow_html=True,
# )
