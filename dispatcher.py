import logging

from Orchestrator.router import route_message      # 0.6B + tools
from Models.qwen8b_VL import call_chat_stream as call_chat_stream_vl  # 8B VL
from Search.ollama_search import chat_with_ollama_web  # 8B + web_search/web_fetch
from ollama import chat
from Tools.system import TOOLS

logger = logging.getLogger("Dispatcher")

# Indices d'écrans mss()
SCREEN_MAIN = 2      # 3440x1440, (0,0)
SCREEN_RIGHT = 1     # 1920x1080, à droite
SCREEN_LEFT = 3      # 1080x1920, à gauche


def infer_screen_index_from_text(text: str) -> int:
    """
    Essaie de deviner quel écran l'utilisateur vise.
    """
    t = text.lower()

    ecran_keywords = {
        "troisième écran": SCREEN_RIGHT,
        "3ème écran": SCREEN_RIGHT,
        "3e écran": SCREEN_RIGHT,

        "deuxième écran": SCREEN_MAIN,
        "2ème écran": SCREEN_MAIN,
        "2e écran": SCREEN_MAIN,
        "main screen": SCREEN_MAIN,

        "premier écran": SCREEN_LEFT,
        "1er écran": SCREEN_LEFT,
        "1er ecran": SCREEN_LEFT,
        "écran de gauche": SCREEN_LEFT,

        "écran de droite": SCREEN_RIGHT,
    }

    for k, idx in ecran_keywords.items():
        if k in t:
            logger.info(f"Mot-clé écran détecté : {k!r} -> écran {idx}")
            return idx

    logger.info(f"[Dispatcher-VL] Aucun mot-clé écran trouvé, fallback écran principal ({SCREEN_MAIN})")
    return SCREEN_MAIN


def dispatch_text(text: str) -> tuple[str, str]:
    """
    Retourne (réponse, mode)
    mode in {"system", "vision", "chat"}
    """
    if not text:
        logger.info("Texte vide, fallback sur chat.")
        return "Je n'ai rien recu a traiter.", "chat"

    lower = text.lower().strip()
    logger.info(f"Nouvelle requete : {repr(text)}")

    system_keywords = [
        "ouvre", "lance", "demarre", "démarre", "start", "launch",
        "youtube", "video", "vidéo",
        "timer", "minuteur", "rappel",
        "volume", "son", "mute", "unmute",
        "arrete le timer", "arrête le timer", "stop le timer","alarme", 
        "reveille moi", "réveille moi", "reveille-moi", "réveille-moi",
    ]

    vision_keywords = [
        "capture", "screenshot", "capture ecran", "capture écran",
        "prends une capture", "prend une capture",
        "regarde l'écran", "regarde l ecran",
        "vois l'écran", "vois l ecran",
        "observe l'écran", "observe l ecran", "écran",
    ]

    # 1) Commandes système
    if any(k in lower for k in system_keywords):
        logger.debug("Mot-cle systeme detecte -> route_message (0.6B + tools)")
        answer = route_message(text)
        logger.debug("Reponse mode=system OK")
        return answer, "system"

    # 2) Vision (écran)
    if any(k in lower for k in vision_keywords):
        screen_idx = infer_screen_index_from_text(text)
        logger.debug(f"Mot-cle vision detecte -> qwen3:8b:vl, ecran={screen_idx}")
        answer = call_chat_stream_vl(text, screen_index=screen_idx)
        logger.debug(f"Answer: {answer!r}")
        return answer, "vision"

    # 3) Chat normal (8B texte + tools web)
    logger.debug("Aucun mot-cle systeme/vision -> qwen3:8b avec tools web_search/web_fetch")
    answer = chat_with_ollama_web(text)
    logger.debug(f"Answer: {answer!r}")
    return answer, "chat"


# ---------------------------------------------------------
# WEB SEARCH (ollama)
# ---------------------------------------------------------
# Voir Search/ollama_search.py