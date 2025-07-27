from bot.model.base_model import ModelSettings


class Phi35Settings(ModelSettings):
    """
    Model settings for Phi-3.5-mini Instruct (Q4_K_M).
    This is the only Phi model retained for financial tasks in this project.
    """

    url = (
        "https://huggingface.co/MaziyarPanahi/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct.Q4_K_M.gguf"
    )
    file_name = "Phi-3.5-mini-instruct.Q4_K_M.gguf"

    config = {
        "n_ctx": 4096,       # Max sequence length
        "n_threads": 8,      # Number of CPU threads
        "n_gpu_layers": 33,  # Number of layers offloaded to GPU
    }

    config_answer = {
        "temperature": 0.7,  # Sampling temperature
        "stop": []           # Stop sequences
    }
