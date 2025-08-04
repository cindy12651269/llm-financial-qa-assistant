import time
from pathlib import Path

from chatbot.bot.client.lama_cpp_client import LamaCppClient
from chatbot.bot.model.model_registry import get_model_settings

if __name__ == "__main__":
    # Define the root and model directory
    root_folder = Path(__file__).resolve().parent.parent.parent.parent
    model_folder = root_folder / "models"
    Path(model_folder).parent.mkdir(parents=True, exist_ok=True)
    
    # Load model settings for financial Q&A
    model_settings = get_model_settings("llama-3.2:3b")
    
    # Initialize LLM client (LLaMA model for financial queries)
    llm = LamaCppClient(model_folder, model_settings)

    # Example financial question prompt
    prompt = "What is the EPS (Earnings Per Share) of Tesla in 2023?"
    print(f"Prompt: {prompt}\n")

    # Run single response generation (non-streaming)
    start_time = time.time()
    output = llm.generate_answer(prompt, max_new_tokens=512)
    print("[Single Response]\n" + output)
    took = time.time() - start_time
    print(f"\n--- Took {took:.2f} seconds ---")
    
    # Run streaming response generation (token by token)
    print("[Streaming Response]")
    start_time = time.time()
    stream = llm.start_answer_iterator_streamer(prompt, max_new_tokens=256)
    for output in stream:
        print(output["choices"][0]["delta"].get("content", ""), end="", flush=True)
    took = time.time() - start_time

    print(f"\n--- Took {took:.2f} seconds ---")
