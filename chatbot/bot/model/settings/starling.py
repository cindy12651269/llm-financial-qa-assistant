from bot.model.base_model import ModelSettings


class StarlingSettings(ModelSettings):
    """
    Settings for Starling LM 7B Beta (Q4_K_M).
    Powerful chat model suitable for financial dialogue and reasoning.
    """

    url = "https://huggingface.co/bartowski/Starling-LM-7B-beta-GGUF/resolve/main/Starling-LM-7B-beta-Q4_K_M.gguf"
    file_name = "Starling-LM-7B-beta-Q4_K_M.gguf"

    config = {
        "n_ctx": 4096,       # Max sequence length supported by model
        "n_threads": 8,      # Number of CPU threads used during inference
        "n_gpu_layers": 50   # Number of transformer layers offloaded to GPU
    }

    config_answer = {
        "temperature": 0.7,  # Sampling temperature for answer creativity
        "stop": []           # Stop tokens (empty = default)
    }

