"""
import logging
import io
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

logger = logging.getLogger("TTS")

# --------------------------------------------------
# Modèle Qwen3-TTS
# --------------------------------------------------
attn_implementation = "flash_attention_2"
MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

model = Qwen3TTSModel.from_pretrained(
    MODEL,
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation=attn_implementation,
)

def synthesize_to_wav_bytes_Qwen3(text):

    logger.info("Synthèse demandée: %r", (text or "")[:80])

    if not text or not text.strip():
        return b""

    # Nettoyage typographique léger
    text = text.replace("’", "'").replace("“", '"').replace("”", '"').strip()
    
    # single inference
    wavs, sr = model.generate_voice_design(
        text,
        language="English",
        instruct="Speak in a gentle, warm voice with a light, joyful tone.",
    )    

    # WAV en mémoire
    buffer = io.BytesIO()
    sf.write(buffer, wavs[0], sr, format="WAV")
    buffer.seek(0)
    wav_bytes = buffer.getvalue()
    return wav_bytes
"""