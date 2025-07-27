import argparse
import sys
import time
from pathlib import Path

from bot.client.lama_cpp_client import LamaCppClient
from bot.model.model_registry import get_model_settings, get_models
from helpers.log import get_logger
from helpers.reader import read_input
from pyfiglet import Figlet
from rich.console import Console
from rich.markdown import Markdown

logger = get_logger(__name__) # Initialize logger


def get_args() -> argparse.Namespace:
    """
    Parse command-line arguments to select model.
    Returns:
        argparse.Namespace: Parsed arguments object
    """
    parser = argparse.ArgumentParser(description="Finance Chatbot")

    model_list = get_models() # Get list of available models
    default_model = model_list[0] # Default to the first model
    
    # Add CLI option to choose model
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

    return parser.parse_args()  # Return parsed arguments


def loop(llm):
    """
    Main loop for user interaction. Renders a welcome message, 
    prompts for questions, sends to LLM, and displays answers.
    Args:
        llm (LamaCppClient): Loaded language model
    """
    console = Console(color_system="windows") # Rich console setup
    custom_fig = Figlet(font="graffiti") # ASCII art title

    # Display welcome message
    console.print(custom_fig.renderText("ChatBot"))
    console.print(
    "[bold magenta]Hi! 👋, I'm your financial assistant. Here to help you. "
    "\nAsk me anything about trading, metrics, or financial reports. Type 'exit' to stop.[/bold magenta]"
    )
    while True:
        console.print("[bold green]Please enter your financial question:[/bold green]")
        question = read_input()

        if question.lower() == "exit": # Exit if user types 'exit'
            break

        start_time = time.time() # Start timing response

        prompt = llm.generate_qa_prompt(question=question) # Format prompt for LLM

        # Show question
        console.print(f"\n[bold green]Question:[/bold green] {question}")
        console.print("\n[bold green]Answer:[/bold green]")
        
        # Stream answer from model and render result
        answer = llm.stream_answer(prompt, max_new_tokens=1000)
        console.print("\n[bold magenta]Formatted Answer:[/bold magenta]")
        console.print(Markdown(answer))

        # Print elapsed time
        took = time.time() - start_time
        print(f"--- Took {took:.2f} seconds ---")


def main(parameters):
    model_settings = get_model_settings(parameters.model) # Get config for selected model
    """
    Load the selected model and launch chatbot loop.
    Args:
        parameters (argparse.Namespace): CLI arguments
    """
    # Construct model directory path
    root_folder = Path(__file__).resolve().parent.parent.parent 
    model_folder = root_folder / "models"
    Path(model_folder).parent.mkdir(parents=True, exist_ok=True) # Ensure path exists

    llm = LamaCppClient(model_folder=model_folder, model_settings=model_settings) # Initialize LLM
    loop(llm) # Launch chatbot loop


if __name__ == "__main__":
    try:
        args = get_args() # Parse CLI arguments
        main(args) # Start application
    except Exception as error:
        logger.error(f"An error occurred: {str(error)}", exc_info=True, stack_info=True)
        sys.exit(1)
