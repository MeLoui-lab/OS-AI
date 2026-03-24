import base64
import io
import logging
import time
import tkinter as tk

from mss import mss
from ollama import chat
from PIL import Image

logger = logging.getLogger("qwen8b_VL")

FLASH_DURATION = 3.0          # secondes
FLASH_COLOR = "#ffffff89"
FLASH_WIDTH = 8

def _flash_overlay(monitor, duration_s=FLASH_DURATION, color="#ffffff", width=FLASH_WIDTH, alpha=0.35):

    x, y, w, h = monitor["left"], monitor["top"], monitor["width"], monitor["height"]

    root = tk.Tk()
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.attributes("-alpha", alpha)

    try:
        root.attributes("-transparentcolor", "magenta")
        bg = "magenta"
    except Exception:
        bg = ""

    root.geometry(f"{w}x{h}+{x}+{y}")
    canvas = tk.Canvas(root, bg=bg, highlightthickness=0)
    canvas.pack(fill="both", expand=True)

    canvas.create_rectangle(
        width / 2,
        width / 2,
        w - width / 2,
        h - width / 2,
        outline=color,
        width=width,
    )

    # On programme la fermeture
    root.after(int(duration_s * 1000), root.destroy)

    # Boucle Tk **bloquante** mais courte
    root.mainloop()


def capture_screen(screen_index: int = 1, flash: bool = True) -> Image.Image:
    with mss() as sct:
        monitors = sct.monitors
        logger.info(f"Monitors détectés :")
        for i, m in enumerate(monitors):
            logger.info(f"  {i} -> {m}")

        if screen_index >= len(monitors):
            raise ValueError(f"Ecran {screen_index} inexistant, trouvés: {len(monitors) - 1}")

        monitor = monitors[screen_index]
        logger.info(f"Capture écran index={screen_index}, zone={monitor}")

        overlay = _flash_overlay(monitor) if flash else None
        if overlay:
            time.sleep(0.05)
        shot = sct.grab(monitor)
        return Image.frombytes("RGB", (shot.width, shot.height), shot.rgb)



def resize_for_vl(img: Image.Image, max_side: int = 1400) -> Image.Image:
    """Redimensionne pour rester sous les limites runtime VL."""
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)


def capture_to_base64(screen_index: int = 1, max_side: int = 1400, flash: bool = True) -> str:
    """Capture, redimensionne et encode l'ecran en base64 PNG."""
    img = resize_for_vl(capture_screen(screen_index, flash=flash), max_side=max_side)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

def call_chat_stream(text: str, screen_index: int = 1) -> str:
    """
    Appelle qwen3:8b VL en streaming avec capture ecran.
    """
    b64_img = capture_to_base64(screen_index)
    stream = chat(
        model="qwen3-vl:8b",  # ajuste si ton tag differe
        messages=[
            {
                "role": "user",
                "content": text,
                "images": [b64_img]
            }
        ],
        stream=True,
    )

    content_parts: list[str] = []
    log_buffer = ""

    logger.info("Debut stream qwen3:8b:vl")

    for chunk in stream:
        msg = chunk.message

        if msg.content:
            # on garde tout pour la reponse finale
            content_parts.append(msg.content)
            log_buffer += msg.content

            # on log seulement quand on a un bout lisible
            if any(p in log_buffer for p in [".", "!", "?", "\n"]) or len(log_buffer) > 80:
                logger.info(f"{log_buffer}")
                log_buffer = ""

    # s'il reste un bout non logge a la fin
    if log_buffer:
        logger.info(f"[8B-VL-content-buffer] {log_buffer}")

    content_full = "".join(content_parts)

    logger.info("Fin stream qwen3:8b:vl")
    logger.info(f"Content complet len={len(content_full)}")

    return content_full
