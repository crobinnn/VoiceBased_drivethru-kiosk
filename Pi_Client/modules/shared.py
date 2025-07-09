# Module untuk general config & utility yg dipake sama modul2 lainnya
import logging
import json
import websockets
import subprocess
import asyncio

# Setup system logging info
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Variabel untuk simpan address inference server & Flag untuk koneksi dengan server
inference_addr = ""
ip_addr = ""
inference_connected = False

# Path audio sound untuk diplay berdasarkan state program dimana
INVALID_INTENT_PATH = "/home/robs/Skripsi/drivethru/backend/audios/invalid_intent.wav"
ITEM_NOT_IN_ORDER_PATH = "/home/robs/Skripsi/drivethru/backend/audios/item_not_in_order.wav"
NOT_IN_ORDER_WITH_VALID = "/home/robs/Skripsi/drivethru/backend/audios/not_inorder_valid.wav"
VALID_PATH = "/home/robs/Skripsi/drivethru/backend/audios/valid_order.wav"
INVALID_PATH = "/home/robs/Skripsi/drivethru/backend/audios/invalid_order.wav"
INVALID_WITH_VALID_PATH = "/home/robs/Skripsi/drivethru/backend/audios/invalid_order_with_validmenu.wav"
FALSE_QTY = "/home/robs/Skripsi/drivethru/backend/audios/false_qty.wav"
INCOMPLETE_CHANGE_PATH = "/home/robs/Skripsi/drivethru/backend/audios/incomplete_change.wav"
WELCOME_PATH = "/home/robs/Skripsi/drivethru/backend/audios/welcome.wav"
DONE_PATH = "/home/robs/Skripsi/drivethru/backend/audios/confirm_order.wav"
CANCEL_PATH = "/home/robs/Skripsi/drivethru/backend/audios/cancel_order.wav"

# Port websocket aplikasi flutter
APP_PORT = 33941

# Setting input audio
PULSE_DEVICE = "echo-cancel-source" # nama input device pulse audio yang sudah diload dengan module echo cancellation pulse
SAMPLERATE_IN = 16000 # samplerate audio in
CHANNELS = 1 #channel mono audio karena webrtcvad hanya support mono audio

# VAD Configuration
VAD_MODE = 1 # sensitivitas VAD
CHUNK_DURATION_MS = 20 # durasi tiap chunk audio (ms) untuk diposes vad
CHUNK_SIZE = int(SAMPLERATE_IN * CHUNK_DURATION_MS / 1000) * 2 # chunk size 640Bytes
SILENCE_TIMEOUT = 1.5 # berapa lama slience terdeteksi untuk stop record
MIN_VOLUME = 2500 # Batas volume minimum untuk dianggap sebagai suara sebagai filter awal selain VAD
CONSECUTIVE_VOICE = 3 # Jumlah chunk harus 3 chunk berturut2 dengan deteksi suara oleh VAD baru dianggap ada voice activity
MAX_RECORDING_DURATION = 20.0 # Max durasi recording per session
COOLDOWN_PERIOD = 0.5 # delay sblm mulai recording baru

# TOF Sensor Configuration
TOF_DETECTION_THRESHOLD = 80 #batas jarak deteksi tof 50cm
TOF_STABLE_DETECTION_TIME = 1.0 #waktu terdeteksi 
TOF_STABLE_UNDETECTION_TIME = 1.0 #waktu gak terdeteksi

connected_app = None # simpan object websocket aplikasi
is_running = True # flag program running/gak

#mapping untuk ubah string angka jadi integer buat tracking dan kirim ke app 
NUMBER_MAPPING = {
    "satu": 1, "dua": 2, "tiga": 3, "empat": 4, "lima": 5,
    "enam": 6, "tujuh": 7, "delapan": 8, "sembilan": 9, "sepuluh": 10, "sebelas": 11
}

#state dari client drive thru
class ClientState:
    WAITING_FOR_CAR = "waiting_for_car" #state awal nunggu mobil
    CAR_DETECTED = "car_detected" #state saat mobil kedetek
    INIT_STATE = "init_state" #state awal interaksi dengan customer
    PROCESS_STATE = "process_state" #state pemrosesan order customer
    CANCELED_STATE = "canceled_state" #state jika dibatalkan pesanan
    CONFIRMED_STATE = "confirmed_state" #state jika pesanan dikonfirmasi
    PROCESSING_ORDER = "processing_order" #state saat sedang memroses order ke server

async def play_sound(audio):
    """Function untuk play audio yang bisa dipanggil modul lainnya"""
    audio_paths = {
        'valid': VALID_PATH,
        'done': DONE_PATH,
        'welcome': WELCOME_PATH,
        'cancel': CANCEL_PATH,
        'invalid_intent': INVALID_INTENT_PATH,
        'item_not_in_order': ITEM_NOT_IN_ORDER_PATH,
        'not_in_order_with_valid': NOT_IN_ORDER_WITH_VALID,
        'invalid': INVALID_PATH,
        'invalid_with_valid': INVALID_WITH_VALID_PATH,
        'incomplete_change': INCOMPLETE_CHANGE_PATH,
        'insufficient_quantity': FALSE_QTY
    }
    
    if audio in audio_paths:
        try:
            # Subprocess command pulse audio play audio file berdasarkan audio path yang dikasi
            play_process = await asyncio.create_subprocess_exec(
                "paplay", audio_paths[audio],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            await play_process.wait()
            print(f"✅ Played {audio} sound")
        except Exception as e:
            print(f"⚠️ Could not play {audio} sound: {e}")
            
async def send_to_app(message):
    """Function untuk kirim pesan ke aplikasi untuk update UI & state"""
    global connected_app
    if connected_app:
        try:
            await connected_app.send(json.dumps(message))
            logger.info(f"Sent to app: {message}")
        except websockets.exceptions.ConnectionClosed:
            logger.warning("App disconnected during send")
        except Exception as e:
            logger.error(f"Error sending to app: {e}")
