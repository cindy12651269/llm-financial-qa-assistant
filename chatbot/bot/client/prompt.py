# Financial-specific system and QA prompt templates for RAG Chatbot
# Corresponds to: All Stories in demo.md (finance-themed chatbot assistant)

# Used in: All Stories (acts as the chatbot's persona)
SYSTEM_TEMPLATE = """You are a knowledgeable and trustworthy financial assistant.
You explain investment concepts, trading terminology, and valuation logic in a clear and concise way.
"""

# Used in: Future expansion with tools (e.g., financial calculator, external APIs)
TOOL_SYSTEM_TEMPLATE = """You are a knowledgeable and trustworthy financial assistant.
You can call tools or functions with the correct input when needed.
You explain financial concepts clearly and cite known terminology.
"""

# Used in: Single-turn questions (e.g., "What is a bid-ask spread?")
# Corresponds to: demo.md > Story 1 - Trading Terminology Primer
QA_PROMPT_TEMPLATE = """Please answer the following finance-related question:
{question}
"""

# Used in: Contextual QA (e.g., "Given this explanation, what is liquidity?")
# Corresponds to: Story 2 - Investment Strategy Concepts, Story 3 - Card Value Estimation
CTX_PROMPT_TEMPLATE = """Relevant information is provided below.
---------------------
{context}
---------------------
Given the context above and not prior knowledge, please answer this finance-related question:
{question}
"""

# Used in: Refinement of answers after additional document chunk retrieved
# Corresponds to: Story 4 - Time-Series & Backtesting (e.g., "Remind me of pitfalls...")
REFINED_CTX_PROMPT_TEMPLATE = """Original Question: {question}
Existing Answer: {existing_answer}
You may improve the answer using the additional context below.
---------------------
{context}
---------------------
If helpful, refine the existing answer. Otherwise, return the original answer.
Refined Answer:
"""

# Used in: Multi-turn conversations (e.g., "What about liquidity?" → rephrase)
# Corresponds to: Rag Chatbot > Story - 1
REFINED_QUESTION_CONVERSATION_AWARENESS_PROMPT_TEMPLATE = """Previous Chat History:
---------------------
{chat_history}
---------------------
Follow-up Question: {question}
Rephrase the question above into a standalone financial query.
Standalone Question:
"""

# Used in: Multi-turn chat (e.g., continuing discussion on Sharpe Ratio or Alpha)
# Corresponds to: Rag Chatbot > Story - 1, and any follow-up in multi-turn flows
REFINED_ANSWER_CONVERSATION_AWARENESS_PROMPT_TEMPLATE = """
You are simulating a professional investment advisor.
Your responses should demonstrate domain knowledge in trading, quantitative finance, and investment strategies.
Chat History:
---------------------
{chat_history}
---------------------
Follow-up Question: {question}
Using the above conversation context, answer the question clearly and accurately.
If context is irrelevant, just answer directly.
Avoid rephrasing the question; give a concise, helpful answer.
"""

def generate_qa_prompt(template: str, system: str, question: str) -> str:
    """
    Generates a prompt for a financial question-answer task.

    Args:
        template (str): A string template with placeholders for system and question.
        system (str): The name or identifier of the financial assistant system.
        question (str): The financial or trading-related question to be included in the prompt.

    Returns:
        str: The generated QA prompt string.
    """
    prompt = template.format(system=system, question=question)
    return prompt


def generate_ctx_prompt(template: str, system: str, question: str, context: str = "") -> str:
    """
    Generates a prompt for a context-aware financial QA task.

    Args:
        template (str): A string template with placeholders for system, question, and context.
        system (str): The name or identifier of the financial assistant system.
        question (str): The financial or investing-related question.
        context (str, optional): Additional relevant context from documents. Defaults to "".

    Returns:
        str: The generated prompt with context included.
    """
    prompt = template.format(system=system, context=context, question=question)
    return prompt

def generate_refined_ctx_prompt(
    template: str, system: str, question: str, existing_answer: str, context: str = ""
) -> str:
    """
    Generates a refined prompt for improving an existing financial answer with new context.

    Args:
        template (str): A string template with placeholders for system, question, existing_answer, and context.
        system (str): The name or identifier of the financial assistant system.
        question (str): The original finance-related question.
        existing_answer (str): The current answer to potentially improve.
        context (str, optional): Newly retrieved context information. Defaults to "".

    Returns:
        str: The updated prompt for refining the answer.
    """
    prompt = template.format(
        system=system,
        context=context,
        existing_answer=existing_answer,
        question=question,
    )
    return prompt


def generate_conversation_awareness_prompt(template: str, system: str, question: str, chat_history: str) -> str:
    """
    Generates a conversation-aware prompt for financial multi-turn QA.

    Args:
        template (str): A string template with placeholders for system, question, and chat_history.
        system (str): The name or identifier of the financial assistant system.
        question (str): The follow-up financial question to answer.
        chat_history (str): The preceding conversation context.

    Returns:
        str: The generated prompt tailored to conversational context.
    """
    prompt = template.format(
        system=system,
        chat_history=chat_history,
        question=question,
    )
    return prompt