from asyncio import get_event_loop
from typing import Any

from entities.document import Document
from helpers.log import get_logger

from bot.client.lama_cpp_client import LamaCppClient
from bot.conversation.chat_history import ChatHistory
from bot.conversation.ctx_strategy import AsyncTreeSummarizationStrategy, BaseSynthesisStrategy

logger = get_logger(__name__)


def refine_question(llm: LamaCppClient, question: str, chat_history: ChatHistory, max_new_tokens: int = 128) -> str:
    """
    Rephrase the user's financial question to make it clearer, using chat history context.

    Args:
        llm (LamaCppClient): The LLM client customized for finance.
        question (str): The original user query.
        chat_history (ChatHistory): The conversation history.
        max_new_tokens (int): Max length for the generated refined question.

    Returns:
        str: The improved, standalone financial question.
    """

    if chat_history:
        logger.info("--- Refining financial question using chat history ---")

        conversation_awareness_prompt = llm.generate_refined_question_conversation_awareness_prompt(
            question, str(chat_history)
        )

        logger.info(f"--- Prompt:\n {conversation_awareness_prompt} \n---")

        refined_question = llm.generate_answer(conversation_awareness_prompt, max_new_tokens=max_new_tokens)

        logger.info(f"--- Refined Question: {refined_question} ---")

        return refined_question
    else:
        return question


def answer(llm: LamaCppClient, question: str, chat_history: ChatHistory, max_new_tokens: int = 512) -> Any:
    """
    Stream a financial domain-specific answer, optionally using prior chat history.

    Args:
        llm (LamaCppClient): The finance-tuned LLM client.
        question (str): The user query.
        chat_history (ChatHistory): Conversation memory.
        max_new_tokens (int): Output token limit.

    Returns:
        Any: Token stream for displaying response progressively.
    """
    if chat_history:
        logger.info("--- Generating financial answer with chat history ---")

        conversation_awareness_prompt = llm.generate_refined_answer_conversation_awareness_prompt(
            question, str(chat_history)
        )

        logger.debug(f"--- Prompt:\n {conversation_awareness_prompt} \n---")

        streamer = llm.start_answer_iterator_streamer(conversation_awareness_prompt, max_new_tokens=max_new_tokens)

        return streamer
    else:
        prompt = llm.generate_qa_prompt(question=question)
        logger.debug(f"--- Prompt:\n {prompt} \n---")
        streamer = llm.start_answer_iterator_streamer(prompt, max_new_tokens=max_new_tokens)
        return streamer


def answer_with_context(
    llm: LamaCppClient,
    ctx_synthesis_strategy: BaseSynthesisStrategy,
    question: str,
    chat_history: ChatHistory,
    retrieved_contents: list[Document],
    max_new_tokens: int = 512,
):
    """
    Generate a financial answer using retrieved context (RAG) and question history.

    Args:
        llm (LamaCppClient): The finance-aware LLM client.
        ctx_synthesis_strategy (BaseSynthesisStrategy): Context summarization strategy.
        question (str): User's financial question.
        chat_history (ChatHistory): Past interactions.
        retrieved_contents (list[Document]): Relevant documents to enhance accuracy.
        max_new_tokens (int): Token generation limit.

    Returns:
        tuple: (streamer, list of formatted prompt strings used).
    """
    if not retrieved_contents:
        return answer(llm, question, chat_history, max_new_tokens=max_new_tokens), []

    if isinstance(ctx_synthesis_strategy, AsyncTreeSummarizationStrategy):
        loop = get_event_loop()
        streamer, fmt_prompts = loop.run_until_complete(
            ctx_synthesis_strategy.generate_response(retrieved_contents, question, max_new_tokens=max_new_tokens)
        )
    else:
        streamer, fmt_prompts = ctx_synthesis_strategy.generate_response(
            retrieved_contents, question, max_new_tokens=max_new_tokens
        )
    return streamer, fmt_prompts
