# qwen8b.py (exemple)

import logging
from ollama import chat

logger = logging.getLogger("qwen8b")


def run_qwen_agent(question: str, tools=None) -> str:
    """
    Agent générique Qwen3:8b avec support des tools Ollama.
    `tools` = liste de fonctions comme [web_search, web_fetch] ou None.
    """
    messages = [
        {"role": "user", "content": question}
    ]

    logger.info("[qwen8b] Nouvelle question (tools=%s): %r", bool(tools), question)

    while True:
        response = chat(
            model="qwen3:8b",
            messages=messages,
            tools=tools,
            think=True,
            stream=True,
        )

        msg = response.message

        if getattr(msg, "thinking", None):
            logger.info("[8B-thinking] %s", msg.thinking)

        if msg.content:
            logger.info("[8B-content-step] %s", msg.content)

        messages.append(msg)

        # Si pas de tools -> réponse finale directe
        if not msg.tool_calls:
            final_content = msg.content or ""
            logger.info("[qwen8b] Réponse finale, len=%d", len(final_content))
            return final_content

        # Sinon: gérer les tool_calls
        from ollama import web_search, web_fetch  # ou injecte ça autrement
        available_tools = {
            "web_search": web_search,
            "web_fetch": web_fetch,
        }

        logger.info("[8B-tools] tool_calls = %s", msg.tool_calls)

        for tool_call in msg.tool_calls:
            func_name = tool_call.function.name
            args = tool_call.function.arguments or {}

            func = available_tools.get(func_name)
            if not func:
                tool_result = f"Tool {func_name} non disponible côté agent."
                logger.warning("[8B-tools] %s", tool_result)
            else:
                try:
                    result = func(**args)
                    tool_result = str(result)
                except Exception as e:
                    tool_result = f"Erreur lors de l'appel de {func_name}: {e}"
                    logger.exception("[8B-tools] %s a planté", func_name)

            truncated = tool_result[:2000 * 4]
            logger.info("[8B-tool-result:%s] %s...", func_name, truncated[:400])

            messages.append(
                {
                    "role": "tool",
                    "tool_name": func_name,
                    "content": truncated,
                }
            )
