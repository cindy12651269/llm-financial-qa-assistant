import argparse
import sys
import time
from pathlib import Path

import streamlit as st
from chatbot.bot.client.lama_cpp_client import LamaCppClient
from chatbot.bot.conversation.chat_history import ChatHistory
from chatbot.bot.conversation.conversation_handler import answer
from chatbot.bot.model.model_registry import get_model_settings, get_models
from helpers.log import get_logger

logger = get_logger(__name__)

# Set page config at the very beginning
st.set_page_config(page_title="Financial Chatbot", page_icon="💰", initial_sidebar_state="collapsed")


@st.cache_resource()
def load_llm(model_name: str, model_folder: Path) -> LamaCppClient:
    """
    Create a LLM session object that points to the model.
    """
    model_settings = get_model_settings(model_name)
    llm = LamaCppClient(model_folder=model_folder, model_settings=model_settings)
    return llm


@st.cache_resource()
def init_chat_history(total_length: int = 2) -> ChatHistory:
    chat_history = ChatHistory(total_length=total_length)
    return chat_history


def init_page(root_folder: Path) -> None:
    left_column, central_column, right_column = st.columns([2, 1, 2])

    with left_column:
        st.write(" ")
    
    # Replaced image with financial-themed AI bot
    with central_column:
     # Display centered finance bot image
     st.image(str(root_folder / "images" / "finance-bot.png"), width=120)

     # Centered title and subtitle using HTML + inline style
     st.markdown(
         """
         <div style='text-align: center; margin-top: 0.5em;'>
             <span style='font-size: 28px; font-weight: bold;'>Your Financial Assistant</span><br/>
             <span style='color: gray;'>Got a financial question? I’m here to help!</span>
         </div>
         """,
         unsafe_allow_html=True,
     )

    with right_column:
        st.write(" ")

    st.sidebar.title("Options")


@st.cache_resource
def init_welcome_message() -> None:
    """Display assistant greeting on first load."""
    with st.chat_message("assistant"):
        # Customized welcome message for financial theme
        st.write("Got a financial question? I’m here to help!")


def reset_chat_history(chat_history: ChatHistory) -> None:
    """
    Initializes the chat history, allowing users to clear the conversation.
    """
    clear_button = st.sidebar.button("🗑️ Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = []
        chat_history.clear()


def display_messages_from_history():
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def main(parameters) -> None:
    root_folder = Path(__file__).resolve().parent.parent
    model_folder = root_folder / "models"
    Path(model_folder).parent.mkdir(parents=True, exist_ok=True)

    model = parameters.model
    max_new_tokens = parameters.max_new_tokens

    init_page(root_folder)
    llm = load_llm(model, model_folder)
    chat_history = init_chat_history(2)
    reset_chat_history(chat_history)
    init_welcome_message()
    display_messages_from_history()

    # Supervise user input
    if user_input := st.chat_input("Input your financial question!"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(user_input)

        # Display assistant response in chat message container
        start_time = time.time()
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for token in answer(llm=llm, question=user_input, chat_history=chat_history, max_new_tokens=max_new_tokens):
                full_response += llm.parse_token(token)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        # Add assistant response to chat history
        chat_history.append(f"question: {user_input}, answer: {full_response}")
        st.session_state.messages.append({"role": "assistant", "content": full_response})

        took = time.time() - start_time
        logger.info(f"\n--- Took {took:.2f} seconds ---")


def get_args() -> argparse.Namespace:
    """Parse CLI arguments for chatbot UI."""
    parser = argparse.ArgumentParser(description="Financial Chatbot")

    model_list = get_models()
    default_model = model_list[0]

    parser.add_argument(
        "--model",
        type=str,
        choices=model_list,
        help=f"Model to be used. Defaults to {default_model}.",
        required=False,
        const=default_model,
        nargs="?",
        default=default_model,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="The maximum number of tokens to generate in the answer. Defaults to 512.",
        required=False,
        default=512,
    )

    return parser.parse_args()


# streamlit run chatbot_app.py
if __name__ == "__main__":
    try:
        args = get_args()
        main(args)
    except Exception as error:
        logger.error(f"An error occurred: {str(error)}", exc_info=True, stack_info=True)
        sys.exit(1)
