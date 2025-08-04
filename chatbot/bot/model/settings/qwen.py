from chatbot.bot.model.base_model import ModelSettings


class Qwen25ThreeSettings(ModelSettings):
    """
    Settings for Qwen2.5 3B Instruct (Q4_K_M).
    A general-purpose assistant model used for financial queries.
    """

    url = "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
    file_name = "qwen2.5-3b-instruct-q4_k_m.gguf"

    config = {
        "n_ctx": 4096,  # Max sequence length
        "n_threads": 8,  # Number of CPU threads
        "n_gpu_layers": 50,  # Number of layers to offload to GPU
    }

    config_answer = {"temperature": 0.7, "top_p": 0.95, "stop": []}


class Qwen25ThreeMathReasoningSettings(ModelSettings):
    """
    Settings for Qwen2.5 Coder 3B (Q6_K_L).
    Optimized for coding, math, and logic-intensive tasks.
    """

    url = "https://huggingface.co/bartowski/Qwen2.5-Coder-3B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-3B-Instruct-Q6_K_L.gguf"
    file_name = "Qwen2.5-Coder-3B-Instruct-Q6_K_L.gguf"

    config = {
        "n_ctx": 4096,
        "n_threads": 8,
        "n_gpu_layers": 50,
    }

    config_answer = {"temperature": 0.7, "top_p": 0.95, "stop": []}
