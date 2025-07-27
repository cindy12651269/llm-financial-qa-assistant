from bot.model.base_model import ModelSettings


class Llama32ThreeSettings(ModelSettings):
    """
    Model settings for Llama 3.2 3B Instruct (Q4_K_M).
    This is the only retained LLaMA model for financial tasks in this project.
    """

    url = "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    file_name = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"

    config = {
        "n_ctx": 4096,       # Max sequence length
        "n_threads": 8,      # Number of CPU threads
        "n_gpu_layers": 50,  # GPU acceleration layers
    }

    config_answer = {
        "temperature": 0.7,  # Sampling temperature
        "stop": []           # Stop sequences (empty means default)
    }

