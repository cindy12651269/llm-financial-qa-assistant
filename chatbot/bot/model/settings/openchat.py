from bot.model.base_model import ModelSettings


class OpenChat36Settings(ModelSettings):
    """
    Model settings for OpenChat 3.6 8B (Q4_K_M).
    This is the only OpenChat model retained for financial tasks in this project.
    """

    url = "https://huggingface.co/bartowski/openchat-3.6-8b-20240522-GGUF/resolve/main/openchat-3.6-8b-20240522-Q4_K_M.gguf"
    file_name = "openchat-3.6-8b-20240522-Q4_K_M.gguf"

    config = {
        "n_ctx": 4096,       # Max sequence length
        "n_threads": 8,      # Number of CPU threads
        "n_gpu_layers": 50,  # GPU acceleration layers
        "flash_attn": False  # Whether to use flash attention
    }

    config_answer = {
        "temperature": 0.7,  # Sampling temperature
        "stop": []           # Stop sequences
    }

