# Search/ollama_search.py

import logging
import os
import requests
from dotenv import load_dotenv
from ollama import chat

logger = logging.getLogger("ollama_search")

# Charger le .env dès le début (OLLAMA_API_KEY y est définie)
load_dotenv()

OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")

if not OLLAMA_API_KEY:
    logger.warning("[ollama_search] OLLAMA_API_KEY manquant ! La recherche web va planter.")


# ---------- OUTILS WEB PERSO (API HTTP DIRECTE) ----------

def web_search(query: str, max_results: int = 5) -> dict:
    """
    Appelle l'API web_search d'Ollama (cloud).
    Retourne le JSON brut.
    """
    if not OLLAMA_API_KEY:
        raise RuntimeError("OLLAMA_API_KEY n'est pas définie")

    url = "https://ollama.com/api/web_search"
    headers = {
        "Authorization": f"Bearer {OLLAMA_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "query": query,
        "max_results": max_results,
    }

    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    logger.info("[web_search] %d résultats pour %r", len(data.get("results", [])), query)
    return data


def web_fetch(url_to_fetch: str) -> dict:
    """
    Appelle l'API web_fetch d'Ollama (cloud).
    """
    if not OLLAMA_API_KEY:
        raise RuntimeError("OLLAMA_API_KEY n'est pas définie")

    url = "https://ollama.com/api/web_fetch"
    headers = {
        "Authorization": f"Bearer {OLLAMA_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "url": url_to_fetch,
    }

    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    logger.info("[web_fetch] fetch %r OK (titre=%r)", url_to_fetch, data.get("title"))
    return data


available_tools = {
    "web_search": web_search,
    "web_fetch": web_fetch,
}


# ---------- BOUCLE AGENT (Qwen3 + tools web) ----------

def chat_with_ollama_web(question: str) -> str:
    """
    Agent Qwen3:8B + outils web_search / web_fetch (API HTTP maison).
    """
    messages = [{"role": "user", "content": question}]
    logger.info("[ollama_search] Nouvelle question avec outils: %r", question)

    while True:
        response = chat(
            model="qwen3:8b",
            messages=messages,
            tools=[web_search, web_fetch],  # nos fonctions perso
            think=True,
        )

        msg = response.message

        # log thinking
        if getattr(msg, "thinking", None):
            logger.info("[8B-thinking] %s", msg.thinking)

        # contenu
        if msg.content:
            logger.info("[8B-content-step] %s", msg.content)

        messages.append(msg)

        # tool calls ?
        if msg.tool_calls:
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

            # on relance un tour de boucle avec les résultats d’outils
            continue

        # pas de nouvel outil → réponse finale
        final_content = msg.content or ""
        logger.info("[ollama_search] Réponse finale len=%d", len(final_content))
        return final_content
