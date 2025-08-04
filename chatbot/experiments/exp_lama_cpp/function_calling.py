import json
from pathlib import Path

from chatbot.bot.client.lama_cpp_client import LamaCppClient
from chatbot.bot.model.model_registry import get_model_settings

# Finance-specific tool 
def get_latest_stock_price(ticker: str, market: str = "NASDAQ") -> str:
    """
    Simulates a stock price lookup for a given company ticker.

    Args:
        ticker (str): Company ticker symbol (e.g., AAPL, MSFT)
        market (str): Exchange market (e.g., NASDAQ, NYSE)

    Returns:
        str: JSON string containing the simulated stock price data.
    """
    dummy_prices = {
        "AAPL": 189.56,
        "TSLA": 254.12,
        "GOOGL": 139.22,
        "MSFT": 343.20,
    }
    price = dummy_prices.get(ticker.upper(), "unknown")
    return json.dumps({"ticker": ticker.upper(), "market": market, "price": price})


def search_financial_news(query: str, max_results: int = 3) -> list[dict[str, str]]:
    """
    Dummy search for financial news articles (simulated).

    Args:
        query (str): Financial search keyword.
        max_results (int): Max number of results.

    Returns:
        list[dict]: List of simulated article metadata.
    """
    return [
        {"title": f"{query} outlook strong in Q4", "url": "https://finance.example.com/q4-outlook"},
        {"title": f"Analysts predict {query} rally", "url": "https://finance.example.com/rally"},
    ]


# Function definitions passed to the LLM as tool metadata 
TOOLS_CONFIG = [
    {
        "type": "function",
        "function": {
            "name": "get_latest_stock_price",
            "description": "Retrieve the latest simulated stock price for a company",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol, e.g. AAPL"},
                    "market": {"type": "string", "enum": ["NASDAQ", "NYSE"]},
                },
                "required": ["ticker"],
            },
        },
    }
]

# Tool map - when functionary chooses a tool, run the corresponding function from this map
TOOLS_MAP = {
    "get_latest_stock_price": get_latest_stock_price,
}

if __name__ == "__main__":
    root_folder = Path(__file__).resolve().parent.parent.parent.parent
    model_folder = root_folder / "models"
    Path(model_folder).parent.mkdir(parents=True, exist_ok=True)

    print(get_latest_stock_price(ticker="AAPL", market="NASDAQ"))
    print(search_financial_news(query="Tesla"))

    model_settings = get_model_settings("llama-3.2:3b")

    llm = LamaCppClient(model_folder, model_settings)

    tools = llm.retrieve_tools(prompt="What is the latest price of AAPL?", tools=TOOLS_CONFIG, tool_choice=None)
    print(tools)

    tools = llm.retrieve_tools(prompt="What is the current price of TSLA stock?", tools=TOOLS_CONFIG)
    print(tools)

    tools = llm.retrieve_tools(
        prompt="Get me the latest stock price of AAPL.",
        max_new_tokens=256,
        tools=TOOLS_CONFIG,
        tool_choice="get_latest_stock_price",
    )
    print(tools)

    if len(tools) > 0:
        function_name = tools[0]["function"]["name"]
        function_args = json.loads(tools[0]["function"]["arguments"])
        func_to_call = TOOLS_MAP.get(function_name, None)
        function_response = func_to_call(**function_args)
        print(f"Tool response: {function_response}")
        prompt_with_function_response = llm.generate_ctx_prompt(
            question="What is the latest price of AAPL?", context=function_response
        )

        stream = llm.start_answer_iterator_streamer(
            prompt=prompt_with_function_response,
            max_new_tokens=256,
        )
        for output in stream:
            print(output["choices"][0]["delta"].get("content", ""), end="", flush=True)
