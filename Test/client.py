import requests, base64, io
import soundfile as sf
import sounddevice as sd

def ask(text: str) -> dict:
    resp = requests.post("http://127.0.0.1:8000/assistant", json={"text": text})
    resp.raise_for_status()
    return resp.json()

def play_from_b64(audio_b64: str):
    audio_bytes = base64.b64decode(audio_b64)          # bytes WAV en mémoire
    data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    sd.play(data, sr, blocking=True)

if __name__ == "__main__":
    payload = ask("Hi, how are you?")
    play_from_b64(payload["audio_b64"])