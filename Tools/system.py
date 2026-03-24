import subprocess
import logging
import webbrowser
import urllib.parse
import re
import requests
import threading
import winsound
import ctypes
from datetime import datetime, timedelta

logger = logging.getLogger("system_tools")

# Constantes Windows pour les touches multimédia
VK_VOLUME_MUTE  = 0xAD
VK_VOLUME_DOWN  = 0xAE
VK_VOLUME_UP    = 0xAF
KEYEVENTF_KEYUP = 0x0002

SOUND_PATH = r"C:\AI\OS_AI\Sounds\Teminite.wav"  # <- son par défaut

user32 = ctypes.WinDLL("user32")

# État global du son (timer / alarm / None)
CURRENT_SOUND_SOURCE: str | None = None


# ---------------------------------------------------------
# APPS
# ---------------------------------------------------------
def open_app(name: str) -> str:
    apps = {
        "zen": r"C:\Program Files\Zen Browser\zen.exe",
        "zen browser": r"C:\Program Files\Zen Browser\zen.exe",
        "notepad": "notepad.exe",
        "explorer": "explorer.exe",
        "file explorer": "explorer.exe",
    }

    key = name.lower().strip()

    aliases = {
        "zenbrowser": "zen browser",
        "zen-browser": "zen browser",
        "bloc notes": "notepad",
        "fileexplorer": "file explorer",
    }
    key = aliases.get(key, key)

    exe = apps.get(key)

    if exe is None:
        return f"Application inconnue : {name}"

    try:
        subprocess.Popen(exe)
        return f"Application '{name}' lancée avec succès."
    except Exception as e:
        return f"Erreur : {e}"


def write_text(text: str) -> str:
    logger.info(f"[Tool] write_text exécuté -> {text}")
    return f"Texte écrit : {text}"


def open_url(url: str) -> str:
    try:
        webbrowser.open(url)
        return f"URL lancée : {url}"
    except Exception as e:
        return f"Erreur : {e}"


# ---------------------------------------------------------
# YOUTUBE
# ---------------------------------------------------------
def open_youtube_video(query: str):
    q = urllib.parse.quote(query)
    url = f"https://www.youtube.com/results?search_query={q}"

    html = requests.get(url).text

    matches = re.findall(
        r'"videoRenderer":\{"videoId":"([a-zA-Z0-9_-]{11})"',
        html
    )

    if not matches:
        return f"Aucune vidéo longue trouvée pour : {query}"

    video_id = matches[0]
    video_url = f"https://www.youtube.com/watch?v={video_id}"

    webbrowser.open(video_url)
    return f"Vidéo lancée : {video_url}"


# ---------------------------------------------------------
# SON COMMUN (timer + alarme)
# ---------------------------------------------------------
def _play_sound_loop(source: str, path: str = SOUND_PATH):
    """
    Joue le son en boucle jusqu'à appel de stop_sound().
    source ∈ {"timer", "alarm"} pour savoir ce qui tourne.
    """
    global CURRENT_SOUND_SOURCE
    CURRENT_SOUND_SOURCE = source

    winsound.PlaySound(
        path,
        winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP
    )


def stop_sound() -> str:
    global CURRENT_SOUND_SOURCE
    try:
        winsound.PlaySound(None, winsound.SND_PURGE)
        if CURRENT_SOUND_SOURCE:
            msg = f"{CURRENT_SOUND_SOURCE.capitalize()} arrêtée."
        else:
            msg = "Aucun son en cours."
        CURRENT_SOUND_SOURCE = None
        return msg
    except Exception as e:
        return f"Erreur lors de l'arrêt du son : {e}"


# ---------------------------------------------------------
# TIMER
# ---------------------------------------------------------
def start_timer(seconds: int):
    """
    Lance un timer non bloquant, puis démarre le son en boucle.
    """
    def timer_end():
        _play_sound_loop("timer", SOUND_PATH)

    t = threading.Timer(seconds, timer_end)
    t.daemon = True
    t.start()

    return f"Timer lancé pour {seconds} secondes."


def stop_timer(*args, **kwargs) -> str:
    """
    Alias pour compatibilité avec le LLM.
    Il peut envoyer seconds=0, on s’en fiche complètement.
    """
    return stop_sound()


# ---------------------------------------------------------
# VOLUME
# ---------------------------------------------------------
def _press_key(vk_code: int):
    user32.keybd_event(vk_code, 0, 0, 0)
    user32.keybd_event(vk_code, 0, KEYEVENTF_KEYUP, 0)


def volume_up(steps: int = 1) -> str:
    for _ in range(max(1, steps)):
        _press_key(VK_VOLUME_UP)
    return f"Volume augmenté ({steps} crans)."


def volume_down(steps: int = 1) -> str:
    for _ in range(max(1, steps)):
        _press_key(VK_VOLUME_DOWN)
    return f"Volume baissé ({steps} crans)."


def volume_mute_unmute() -> str:
    _press_key(VK_VOLUME_MUTE)
    return "Mute basculé."


# ---------------------------------------------------------
# ALARME
# ---------------------------------------------------------
def _parse_alarm_delay_seconds(time_str: str) -> int:
    s = time_str.strip().lower().replace(" ", "")
    s = s.replace("h", ":")

    m = re.match(r"^(\d{1,2})(?::(\d{1,2}))?$", s)
    if not m:
        raise ValueError(f"Heure invalide: {time_str}")

    hour = int(m.group(1))
    minute = int(m.group(2) or 0)

    now = datetime.now()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

    if target <= now:
        target += timedelta(days=1)

    delta = target - now
    return int(delta.total_seconds())


def set_alarm(time: str) -> str:
    """
    Un SEUL paramètre : time (ex : "20h50", "08:30").
    """
    try:
        delay = _parse_alarm_delay_seconds(time)
    except ValueError as e:
        return f"Heure d'alarme invalide : {time} ({e})"

    def alarm_end():
        _play_sound_loop("alarm", SOUND_PATH)

    t = threading.Timer(delay, alarm_end)
    t.daemon = True
    t.start()

    minutes = delay // 60
    return f"Alarme programmée pour {time} (~dans {minutes} minutes)."


# ---------------------------------------------------------
# TOOLS EXPOSÉS AU LLM
# ---------------------------------------------------------
TOOLS = {
    "open_app": open_app,
    "write_text": write_text,
    "open_youtube_video": open_youtube_video,
    "start_timer": start_timer,
    "stop_sound": stop_sound,          # <- nouvelle API propre
    "stop_timer": stop_timer,          # <- alias, au cas où
    "volume_up": volume_up,
    "volume_down": volume_down,
    "volume_mute_unmute": volume_mute_unmute,
    "set_alarm": set_alarm,
}
