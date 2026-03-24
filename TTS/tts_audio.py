import wave
import simpleaudio as sa
import io

def wav_bytes_to_audio(wav_bytes):
    # Lecture immédiate depuis les bytes
    buf_play = io.BytesIO(wav_bytes)
    with wave.open(buf_play, "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        nchannels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()

    play_obj = sa.play_buffer(frames, nchannels, sampwidth, framerate)
    play_obj.wait_done()