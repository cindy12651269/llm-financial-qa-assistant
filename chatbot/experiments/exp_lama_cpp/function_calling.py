import json
from pathlib import Path

from chatbot.bot.client.lama_cpp_client import LamaCppClient
from chatbot.bot.model.model_registry import get_model_settings
from chatbot.financial_fetcher import (
    get_stock_price,
    get_financial_news,
    get_financial_metric
)

# Tool metadata passed to the LLM
tools_config = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Fetch the latest stock price using fallback logic (Alpha Vantage or Yahoo).",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol, e.g. AAPL, TSLA"
                    },
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_financial_news",
            "description": "Fetch recent financial news headlines for a company or topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keyword or company name, e.g. Tesla, earnings"
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_financial_metric",
            "description": "Fetch a financial metric such as EPS, revenue, or net income for a given ticker, year, and optional quarter.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker, e.g. TSLA"},
                    "year": {"type": "integer", "description": "Fiscal year, e.g. 2023"},
                    "quarter": {"type": "integer", "description": "Quarter (1-4), optional."},
                    "metric": {"type": "string", "description": "Financial metric, e.g. 'eps', 'revenue', 'net income'"}
                },
                "required": ["ticker", "year", "metric"]
            },
        },
    }
]

# Mapping from function name to Python callable
tools_map = {
    "get_stock_price": get_stock_price,
    "get_financial_news": get_financial_news,
    "get_financial_metric": get_financial_metric
}

if __name__ == "__main__":
    # Define root and model folder paths
    root_folder = Path(__file__).resolve().parent.parent.parent.parent
    model_folder = root_folder / "models"
    Path(model_folder).parent.mkdir(parents=True, exist_ok=True)

    # Load model and tool
    model_settings = get_model_settings("llama-3.2:3b")
    llm = LamaCppClient(model_folder, model_settings)

    # Example direct call
    print("--- Stock Price ---")
    print(get_stock_price("AAPL"))

    print("--- Financial News ---")
    news = get_financial_news("Tesla")
    for item in news:
        print(f"- {item['title']} ({item['url']})")

    print("--- Financial Metric ---")
    print(get_financial_metric("TSLA", 2023, 4, "eps"))

    # Simulate user tool request
    user_prompt = "What is the latest price of AAPL?"
    tools = llm.retrieve_tools(prompt=user_prompt, tools=tools_config)
    print("\nTools selected by model:")
    print(tools)

    # Execute selected tool if available
    if tools:
        tool_call = tools[0]
        fn_name = tool_call["function"]["name"]
        fn_args = json.loads(tool_call["function"]["arguments"])
        fn = tools_map.get(fn_name)

        if fn:
            result = fn(**fn_args)
            print("\nTool result:")
            print(result)

            # Final answer generation
            prompt_with_context = llm.generate_ctx_prompt(
                question=user_prompt,
                context=result
            )

            print("\nGenerated Answer:")
            stream = llm.start_answer_iterator_streamer(prompt_with_context, max_new_tokens=256)
            for output in stream:
                print(output["choices"][0]["delta"].get("content", ""), end="", flush=True)
