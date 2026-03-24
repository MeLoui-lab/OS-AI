# main.py
import logging
import base64
from fastapi import FastAPI
from pydantic import BaseModel
#from tts_Qwen3 import synthesize_to_wav_bytes_Qwen3
from LuxTTS.tts_Lux import synthesize_to_wav_bytes_Lux
from dispatcher import dispatch_text
from tts_audio import wav_bytes_to_audio
from fastapi.responses import Response

# ----------------------------------------------------------------
# LOGGING GLOBAL
# ----------------------------------------------------------------
logging.basicConfig(level=logging.DEBUG, format="[{asctime}] [{levelname:^7}] [{name}] {message}", datefmt="%H:%M:%S", style="{", force=True)

logger = logging.getLogger("main")
logger.info("[main] Logging initialisé.")

# ----------------------------------------------------------------
# FastAPI
# ----------------------------------------------------------------
app = FastAPI()

class Input(BaseModel):
    text: str | None = None
    image: str | None = None  # plus tard

# ----------------------------------------------------------------
# Endpoint /assistant
# ----------------------------------------------------------------
@app.post("/assistant")
def assistant(input: Input):
    text = input.text or ""
    logger.info(f"[API] /assistant appelé avec texte={repr(text)}")
    response, mode = dispatch_text(text)
    logger.info(f"[API] Réponse générée (mode={mode}), génération audio...")
    wav_bytes = synthesize_to_wav_bytes_Lux(response) 
    logger.info(f"[API] Audio généré.")


    return {
    "response": response,
    "mode": mode,
    "audio_b64": base64.b64encode(wav_bytes).decode("ascii"),
    "audio_mime": "audio/wav",
    }
