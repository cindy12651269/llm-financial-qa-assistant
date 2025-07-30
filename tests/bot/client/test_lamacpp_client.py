import asyncio
from unittest.mock import patch
from pathlib import Path

import pytest
from chatbot.bot.client.lama_cpp_client import LamaCppClient
from chatbot.bot.model.model_registry import Model, get_model_settings


@pytest.fixture
def cpu_config():
    config = {
        "n_ctx": 512,
        "n_threads": 2,
        "n_gpu_layers": 0,
    }
    return config


@pytest.fixture
def model_settings():
    model_setting = get_model_settings(Model.LLAMA_3_2_3B.value)
    return model_setting


@pytest.fixture
def lamacpp_client(model_settings, cpu_config):
    with patch.object(model_settings, "config", cpu_config):
        return LamaCppClient(Path("models"), model_settings)


def test_generate_answer(lamacpp_client):
    prompt = "What is the Sharpe Ratio and how is it calculated?"
    generated_answer = lamacpp_client.generate_answer(prompt, max_new_tokens=50)
    assert any(k in generated_answer.lower() for k in ["sharpe", "risk", "investment", "return"])

def test_generate_stream_answer(lamacpp_client):
    prompt = "What does Alpha mean in investment?"
    generated_answer = lamacpp_client.stream_answer(prompt, max_new_tokens=50)
    assert any (k in generated_answer.lower()for k in["alpha", "benchmark", "excess return"])

def test_start_answer_iterator_streamer(lamacpp_client):
    prompt = "What is Beta in portfolio theory?"
    stream = lamacpp_client.start_answer_iterator_streamer(prompt, max_new_tokens=50)
    generated_answer = ""
    for output in stream:
        generated_answer += output["choices"][0]["delta"].get("content", "")
    assert "beta" in generated_answer.lower()


def test_parse_token(lamacpp_client):
    prompt = "Explain risk-adjusted return."
    stream = lamacpp_client.start_answer_iterator_streamer(prompt, max_new_tokens=50)
    generated_answer = ""
    for output in stream:
        generated_answer += lamacpp_client.parse_token(output)
    assert any("risk" in generated_answer.lower() and kw in generated_answer.lower()
           for kw in ["return", "returns", "rate of return"])

@pytest.mark.asyncio
async def test_async_generate_answer(lamacpp_client):
    prompt = "What does a high P/E ratio indicate?"
    task = lamacpp_client.async_generate_answer(prompt, max_new_tokens=50)
    generated_answer = await asyncio.gather(task)
    output = generated_answer[0].lower()

    print("Model Output:", output)  

    expected_keywords = ["valuation", "expensive", "growth", "investor", "overvalued", "price"]
    assert any(k in output for k in expected_keywords), f"Expected one of {expected_keywords} in answer, but got: {output}"

@pytest.mark.asyncio
async def test_async_start_answer_iterator_streamer(lamacpp_client):
    prompt = "What is Free Cash Flow (FCF)?"
    task = lamacpp_client.async_start_answer_iterator_streamer(prompt, max_new_tokens=50)
    stream = await asyncio.gather(task)
    generated_answer = ""
    for output in stream[0]:
        generated_answer += output["choices"][0]["delta"].get("content", "")
    assert "cash" in generated_answer.lower()
