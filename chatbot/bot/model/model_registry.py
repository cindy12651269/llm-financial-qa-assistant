from enum import Enum

from chatbot.bot.model.settings.llama import Llama32ThreeSettings
from chatbot.bot.model.settings.openchat import OpenChat36Settings
from chatbot.bot.model.settings.phi import Phi35Settings
from chatbot.bot.model.settings.qwen import Qwen25ThreeMathReasoningSettings, Qwen25ThreeSettings
from chatbot.bot.model.settings.starling import StarlingSettings


class Model(Enum):
    LLAMA_3_2_3B = "llama-3.2:3b"
    OPENCHAT_3_6_8B = "openchat-3.6"
    PHI_3_5_MINI = "phi-3.5"
    QWEN_2_5_3B = "qwen-2.5:3b"
    QWEN_2_5_CODER_3B = "qwen-2.5:3b-math-reasoning"
    STARLING_LM_7B_BETA = "starling"


SUPPORTED_MODELS = {
    Model.LLAMA_3_2_3B.value: Llama32ThreeSettings,
    Model.OPENCHAT_3_6_8B.value: OpenChat36Settings,
    Model.PHI_3_5_MINI.value: Phi35Settings,
    Model.QWEN_2_5_3B.value: Qwen25ThreeSettings,
    Model.QWEN_2_5_CODER_3B.value: Qwen25ThreeMathReasoningSettings,
    Model.STARLING_LM_7B_BETA.value: StarlingSettings,
}


def get_models():
    return list(SUPPORTED_MODELS.keys())


def get_model_settings(model_name: str):
    model_settings = SUPPORTED_MODELS.get(model_name)

    # validate input
    if model_settings is None:
        raise KeyError(model_name + " is a not supported model")

    return model_settings
