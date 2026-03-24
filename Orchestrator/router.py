import logging
from ollama import chat
from Tools.system import TOOLS

# ---------------------------------------------------------
# LOGGING
# ---------------------------------------------------------
logger = logging.getLogger("router")


# ---------------------------------------------------------
# AGENT LOOP (0.6B + tools) AVEC STREAM & BUFFER
# ---------------------------------------------------------
def route_message(user_text: str):
    """
    Utilise qwen3:0.6b avec tool-calling.
    - stream=True → on reçoit des chunks
    - on bufferise pour les logs
    - on boucle tant qu'il y a des tool_calls
    """

    messages = [
        {
            "role": "system",
            "content": """
Tu contrôles l'ordinateur de l'utilisateur via des outils.

Règles obligatoires :

1. Si l'utilisateur dit "ouvre le bloc notes", "lance zen", "démarre", "start", "launch", appelle `open_app`.
2. Si l'utilisateur prononce les mots "vidéo", "youtube" alors, appelle open_youtube_video(query="<truc>")
    - "cherche <truc> sur youtube" → open_youtube_video(query="<truc>")
    - "lance une vidéo de <truc>" → open_youtube_video(query="<truc>")
    - "mets <musique>" → open_youtube_video(query="<musique>")
3. Si l'utilisateur dit "mets un timer", "lancer un timer" alors, appelle start_timer(<temps>)
    - "Mets un timer de <10> minutes" → start_timer(600)
    - "Mets un timer de <30> secondes" → start_timer(30)
4. Si l'utilisateur dit "arrête le timer", "stop le timer", "éteins l'alarme" alors, appelle stop_sound.
5. Si l'utilisateur dit "monte le son", "augmente le volume", "augmente le son" appelle volume_up(steps=<nombre>).
    - "monte le volume de 5 crans" → volume_up(steps=5)
6. Si l'utilisateur dit "baisse le volume", "diminue le son" appelle volume_down(steps=<nombre>).
    - "baisse le volume de 3 crans" → volume_down(steps=3)
7. Si l'utilisateur dit "mute", "éteint le son", "unmute", "remet le son" appelle volume_mute_unmute().
8. Si l'utilisateur dit "mets une alarme", "réveille-moi à", "réveille moi à", "alarme à" alors, appelle set_alarm(time="<heure>")
    - "Mets une alarme à 7h30" → set_alarm(time="7h30")
    - "Réveille-moi à 20:00" → set_alarm(time="20:00")
Réponds en français, de façon concise après les actions.
            """
        },
        {"role": "user", "content": user_text}
    ]

    while True:
        logger.info("[Router] Nouvel appel qwen3:0.6b (stream=True)")

        # Appel streamé
        stream = chat(
            model="qwen3:1.7b",
            messages=messages,
            tools=list(TOOLS.values()),
            think=True,
            stream=True,
        )

        thinking_parts: list[str] = []
        content_parts: list[str] = []
        tool_calls: list = []

        think_buffer = ""
        content_buffer = ""

        for chunk in stream:
            msg = chunk.message

            # ---- réflexion interne (<think>) ----
            if msg.thinking:
                thinking_parts.append(msg.thinking)
                think_buffer += msg.thinking

                if any(p in think_buffer for p in [".", "!", "?", "\n"]) or len(think_buffer) > 120:
                    logger.info(f"[0.6B-thinking-buffer] {think_buffer}")
                    think_buffer = ""

            # ---- texte utilisateur visible ----
            if msg.content:
                content_parts.append(msg.content)
                content_buffer += msg.content

                if any(p in content_buffer for p in [".", "!", "?", "\n"]) or len(content_buffer) > 80:
                    logger.info(f"[1.7B-content-buffer] {content_buffer}")
                    content_buffer = ""

            # ---- tool calls ----
            if msg.tool_calls:
                tool_calls.extend(msg.tool_calls)

        # vider les buffers restants
        if think_buffer:
            logger.info(f"[1.7B-thinking-buffer] {think_buffer}")
        if content_buffer:
            logger.info(f"[1.7B-content-buffer] {content_buffer}")

        thinking_full = "".join(thinking_parts)
        content_full = "".join(content_parts)

        logger.info("---- Modèle 1.7B (tour complet) ----")
        logger.info(f"Thinking len={len(thinking_full)}")
        logger.info(f"Content complet: {content_full!r}")
        logger.info(f"Tools: {tool_calls}")

        # On ajoute la réponse de l'IA à l'historique de la conversation
        messages.append({
            "role": "assistant",
            "thinking": thinking_full,
            "content": content_full,
            "tool_calls": tool_calls,
        })

        # Pas de tool_calls → réponse finale
        if not tool_calls:
            return content_full

        # Sinon : exécuter les outils puis reboucler
        for call in tool_calls:
            tool_name = call.function.name
            args = call.function.arguments

            if tool_name in TOOLS:
                logger.info(f"[Router] Exécution tool {tool_name} args={args}")
                try:
                    result = TOOLS[tool_name](**args)
                except Exception as e:
                    result = f"Erreur lors de l'exécution de {tool_name}: {e}"
            else:
                result = f"Unknown tool '{tool_name}'"

            messages.append({
                "role": "tool",
                "tool_name": tool_name,
                "content": str(result),
            })
