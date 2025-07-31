import argparse
import sys
import time
from pathlib import Path

from bot.client.lama_cpp_client import LamaCppClient
from bot.conversation.chat_history import ChatHistory
from bot.conversation.conversation_handler import answer_with_context, refine_question
from bot.conversation.ctx_strategy import get_ctx_synthesis_strategies, get_ctx_synthesis_strategy
from bot.memory.embedder import Embedder
from bot.memory.vector_database.chroma import Chroma
from bot.model.model_registry import Model, get_model_settings, get_models
from helpers.log import get_logger
from helpers.prettier import prettify_source
from helpers.reader import read_input
from pyfiglet import Figlet
from rich.console import Console
from rich.markdown import Markdown

logger = get_logger(__name__)  # Initialize the logger for this module


def get_args() -> argparse.Namespace:
    """
    Parses command-line arguments for model selection, token limits, and context strategy.
    Returns an argparse.Namespace object with the parsed options.
    """
    parser = argparse.ArgumentParser(description="Finance AI Assistant")

    model_list = get_models()
    default_model = Model.LLAMA_3_2_3B.value
    # LLaMA 3.2 3B is fast, lightweight, and sufficient for most financial glossary and definition tasks.
    # Suitable as default during development and CLI use.

    synthesis_strategy_list = get_ctx_synthesis_strategies()  # Retrieve all synthesis strategies
    default_synthesis_strategy = synthesis_strategy_list[0]  # Use the first one as default

    # Model argument
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
    # Strategy argument
    parser.add_argument(
        "--synthesis-strategy",
        type=str,
        choices=synthesis_strategy_list,
        help=f"Model to be used. Defaults to {default_synthesis_strategy}.",
        required=False,
        const=default_synthesis_strategy,
        nargs="?",
        default=default_synthesis_strategy,
    )
    # How many chunks to retrieve
    parser.add_argument(
        "--k",
        type=int,
        help="Number of chunks to return from the similarity search. Defaults to 2.",
        required=False,
        default=2,
    )
    # Token limit for LLM generation
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="The maximum number of new tokens to generate.. Defaults to 512.",
        required=False,
        default=512,
    )
    # Return the parsed arguments
    return parser.parse_args()


def loop(llm, chat_history, synthesis_strategy, index, parameters) -> None:
    """
    Main chat loop. Accepts user questions, performs retrieval, and generates answers using LLM.
    """
    custom_fig = Figlet(font="graffiti")  # Create ASCII-styled banner
    console = Console(color_system="windows")  # Initialize rich console for pretty output
    console.print(custom_fig.renderText("ChatBot"))  # Display bot name in ASCII art
    console.print(
        "[bold magenta]Hi! 👋, I'm your financial assistant. Here to help you. "
        "\nAsk me anything about trading, metrics, or financial reports. Type 'exit' to stop.[/bold magenta]"
    )

    while True:
        console.print("[bold green]Please enter your financial question:[/bold green]")
        question = read_input()

        if question.lower() == "exit":
            break

        logger.info(f"--- Question: {question}, Chat_history: {chat_history} ---")  # Log question and chat history

        start_time = time.time()  # Start timer to measure response time
        refined_question = refine_question(llm, question, chat_history)

        # Prepend role instruction to the query for improved LLM focus
        retrieved_contents, sources = index.similarity_search_with_threshold(query=refined_question, k=parameters.k)

        console.print("\n[bold magenta]Sources:[/bold magenta]")
        for source in sources:
            console.print(Markdown(prettify_source(source)))  # Display each source in markdown format

        console.print("\n[bold magenta]Answer:[/bold magenta]")  # Print section header

        # Stream response from LLM using retrieved context
        streamer, fmt_prompts = answer_with_context(
            llm=llm,
            ctx_synthesis_strategy=synthesis_strategy,
            question=refined_question,
            chat_history=chat_history,
            retrieved_contents=retrieved_contents,
            max_new_tokens=parameters.max_new_tokens,
        )
        answer = ""  # Initialize empty answer buffer
        for token in streamer:  # Iterate through streamed tokens
            parsed_token = llm.parse_token(token)  # Parse each token from LLM
            answer += parsed_token
            print(parsed_token, end="", flush=True)

        # Save this Q&A to chat history
        chat_history.append(
            f"question: {refined_question}, answer: {answer}",
        )

        console.print("\n[bold magenta]Formatted Answer:[/bold magenta]")
        if answer:
            console.print(Markdown(answer))
            took = time.time() - start_time
            print(f"\n--- Took {took:.2f} seconds ---")  # Print how long it took
        else:
            console.print("[bold red]Something went wrong![/bold red]")  # Error fallback


def main(parameters):
    """
    Main function to set up model, vector database, and begin the chatbot loop.
    """
    model_settings = get_model_settings(parameters.model)  # Get model configuration based on CLI input

    root_folder = Path(__file__).resolve().parent.parent.parent  # Compute root directory of the project
    model_folder = root_folder / "models"  # Path to local model files
    vector_store_path = root_folder / "vector_store" / "docs_index"  # Path to vector DB

    llm = LamaCppClient(model_folder=model_folder, model_settings=model_settings)  # Initialize LLM client
    synthesis_strategy = get_ctx_synthesis_strategy(parameters.synthesis_strategy, llm=llm)  # Get retrieval strategy
    chat_history = ChatHistory(total_length=2)  # Create chat history manager, only store last 2 exchanges

    embedding = Embedder()  # Create text embedder
    index = Chroma(persist_directory=str(vector_store_path), embedding=embedding)  # Load vector database

    loop(llm, chat_history, synthesis_strategy, index, parameters)  # Enter chat loop


if __name__ == "__main__":
    try:
        args = get_args()  # Parse arguments from command-line
        main(args)  # Run main program
    except Exception as error:
        logger.error(f"An error occurred: {str(error)}", exc_info=True, stack_info=True)
        sys.exit(1)
