import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import asyncio
import socket
import websockets
import json
import numpy as np
from collections import deque
from threading import Lock
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torchcrf import CRF
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== NLP Model Setup ==========
# Parameter dari training mapping slot dan intent ke id karena prediksi model keluarannya dalam bentuk angka
slot2id = {'O': 0, 'B-MENU': 1, 'B-QTY': 2, 'I-MENU': 3, 'I-QTY': 4}
id2slot = {v: k for k, v in slot2id.items()}
intent2id = {'add_item': 0, 'cancel_order': 1, 'change_item': 2, 'confirm_order': 3, 'oos': 4, 'remove_item': 5}
id2intent = {v: k for k, v in intent2id.items()}


digit_to_indonesian = {
    "1": "satu",
    "2": "dua",
    "3": "tiga",
    "4": "empat",
    "5": "lima",
    "6": "enam",
    "7": "tujuh",
    "8": "delapan",
    "9": "sembilan",
    "10": "sepuluh",
    "11": "sebelas"
}

menu = {
    "sosis",
    "es krim",
    "es jeruk",
    "ayam",
    "roti"
}

def normalize_numbers_in_transcription(text):
    """Replace Indonesian number words with digits in transcription"""
    normalized_text = text
    
    # Replace longer phrases first (e.g., "dua belas" before "dua")
    # Sort by length descending to handle multi-word numbers first
    for indonesian_num in sorted(digit_to_indonesian.keys(), key=len, reverse=True):
        digit = digit_to_indonesian[indonesian_num]
        # Use word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(indonesian_num) + r'\b'
        normalized_text = re.sub(pattern, digit, normalized_text, flags=re.IGNORECASE)
    
    return normalized_text

class JointXLMR(nn.Module):
    """
    Class untuk loading model NLP & configure dia untuk multi task intent classification & NER 
    Parameter2 disesuaikan agar sama dengan saat fine tuning
    """
    def __init__(self, model_name, num_intents, num_slots):
        super().__init__()
        # Load model pretrained XLM-R & Parameter2nya sesuaikan sama training
        self.model = AutoModel.from_pretrained(model_name)
        self.log_vars = nn.Parameter(torch.zeros(2)) 
        self.dropout = nn.Dropout(0.1) 
        self.intent_classifier = nn.Linear(self.model.config.hidden_size, num_intents)
        self.slot_classifier = nn.Linear(self.model.config.hidden_size, num_slots)
        self.crf = CRF(num_slots, batch_first=True)
    # Forward pass -> definisikan flow data ke dalam model sesuaikan juga sama training
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        hidden_states = self.dropout(outputs.last_hidden_state)

        # Intent classification
        intent_logits = self.intent_classifier(hidden_states[:, 0, :])

        # Slot classification 
        slot_logits = self.slot_classifier(hidden_states)

        slot_preds = self.crf.decode(slot_logits)
        return {
            "intent_logits": intent_logits,
            "slot_logits": slot_logits,
            "slot_preds": slot_preds
        }

#class untuk melakukan process inference
class NLPProcessor:
    """
    Inference processor dr class model JointXLMR sblmnya 
    
    Load model dengan parameter2nya dari JointXLMR tadi
    - Process text input 
    - Melakukan prediksi intensi dengan nilai confidence
    - Melakukan NER untuk labelling token mana yang merupakan menu dan qty
    - Parsing untuk melakukan mapping menu dengan qty 
    """
    def __init__(self):
        # Select device yaitu GPU (cuda) jika ada
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing NLP model on {self.device}")
        # Load model ke dalam GPU
        self.model = JointXLMR("xlm-roberta-base", len(intent2id), len(slot2id)).to(self.device)
        # Load parameter (weight) & tokenizer model dari hasil fine tuning
        self.model.load_state_dict(torch.load("final_joint_model128_v7.pt", map_location=self.device))
        self.model.eval() 
        self.tokenizer = AutoTokenizer.from_pretrained("./final_joint_model128_v7")
    
    # Function parsing untuk mapping menu dengan qty nya
    def parse_slots(self, tokens, slot_labels):
        digit_strings = {
            "satu", "dua", "tiga", "empat", "lima",
            "enam", "tujuh", "delapan", "sembilan", "sepuluh", "sebelas"
        }

        # Convert tokenization from subword to word-by-word tokenization
        words = []
        current_word = []
        current_slots = []

        for token, slot in zip(tokens, slot_labels):
            if token.startswith("▁"):
                if current_word:
                    words.append(("".join(current_word), current_slots[0]))
                current_word = [token[1:]]
                current_slots = [slot]
            else:
                current_word.append(token)
                current_slots.append(slot)

        if current_word:
            words.append(("".join(current_word), current_slots[0]))

        logger.info(f"Reconstructed words with slots (first 15): {words[:15]}")

        def weighted_distance(pos1, pos2):
            step = 1 if pos2 > pos1 else -1
            distance = 0
            for i in range(pos1 + step, pos2 + step, step):
                token = words[i][0]
                distance += 2 if token == "," else 1
            return abs(distance)

        # Extract quantity spans
        qty_spans = []
        i = 0
        while i < len(words):
            word, slot = words[i]
            if slot == "B-QTY":
                qty_words = [word]
                start = i
                i += 1
                while i < len(words) and words[i][1] == "I-QTY":
                    qty_words.append(words[i][0])
                    i += 1
                full_qty_text = " ".join(qty_words)
                matched_qty = max(
                    (digit for digit in digit_strings if digit in full_qty_text),
                    key=len,
                    default=None
                )
                if matched_qty:
                    qty_spans.append((start, i - 1, matched_qty))
            else:
                i += 1

        logger.info(f"Detected quantity spans: {qty_spans}")

        # Extract menu spans - Modified to track both detected and matched menus
        menu_spans = []
        invalid_menus = []  # Track invalid menus
        i = 0
        while i < len(words):
            word, slot = words[i]
            if slot == "B-MENU":
                start = i
                menu_words = [word]
                i += 1
                while i < len(words) and words[i][1] == "I-MENU":
                    menu_words.append(words[i][0])
                    i += 1
                full_menu_text = " ".join(menu_words)
                full_menu_text_lower = full_menu_text.lower()
                
                # Check if detected menu matches any valid menu
                matched_menu = max(
                    (item for item in menu if item.lower() in full_menu_text_lower),
                    key=len,
                    default=None
                )
                
                if matched_menu:
                    menu_spans.append((start, i - 1, matched_menu))
                else:
                    # Store invalid menu for reporting
                    invalid_menus.append(full_menu_text)
                    logger.info(f"Invalid menu detected: '{full_menu_text}'")
            else:
                i += 1

        logger.info(f"Detected menu spans: {menu_spans}")
        logger.info(f"Invalid menus: {invalid_menus}")

        # NEW ROBUST PARSING LOGIC
        # Create all possible menu-quantity pairs with their distances
        menu_qty_pairs = []
        
        # For each quantity, find all menus and calculate distances
        for qty_idx, (qty_start, qty_end, qty_text) in enumerate(qty_spans):
            qty_center = (qty_start + qty_end) / 2
            
            for menu_idx, (menu_start, menu_end, menu_text) in enumerate(menu_spans):
                menu_center = (menu_start + menu_end) / 2
                
                # Calculate distance between quantity and menu centers
                distance = abs(qty_center - menu_center)
                
                menu_qty_pairs.append({
                    'qty_idx': qty_idx,
                    'menu_idx': menu_idx,
                    'qty_text': qty_text,
                    'menu_text': menu_text,
                    'distance': distance
                })
        
        # Sort pairs by distance (closest first)
        menu_qty_pairs.sort(key=lambda x: x['distance'])
        
        # Assign quantities to menus (greedy approach - closest pairs first)
        used_qty_indices = set()
        used_menu_indices = set()
        results = []
        
        # Process closest pairs first
        for pair in menu_qty_pairs:
            if pair['qty_idx'] not in used_qty_indices and pair['menu_idx'] not in used_menu_indices:
                results.append((pair['qty_text'], pair['menu_text']))
                used_qty_indices.add(pair['qty_idx'])
                used_menu_indices.add(pair['menu_idx'])
        
        # Add remaining menus without quantities (default to "satu")
        for menu_idx, (menu_start, menu_end, menu_text) in enumerate(menu_spans):
            if menu_idx not in used_menu_indices:
                results.append(("satu", menu_text))
        
        # Sort results by original menu position to maintain order
        menu_positions = {menu_text: menu_start for menu_start, menu_end, menu_text in menu_spans}
        results.sort(key=lambda x: menu_positions[x[1]])
        
        # Return both valid results and invalid menus
        return results, invalid_menus

    # Process input text ke dalam model untuk inference
    def process_text(self, text):
        try:
            logger.info(f"Processing text input: '{text}'")
            # Teks ditokenisasi terlebih dahulu dengan tokenizer dari hasil fine tuning dan dikembalikan dalam bentuk tensor
            tokenized = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            
            # Tensor dipindahkan ke dalam GPU yang dimana sudah diload model sblmnya
            input_ids = tokenized["input_ids"].to(self.device)
            attention_mask = tokenized["attention_mask"].to(self.device)

            # Matikan gradient computation karena ini hanya untuk inference
            with torch.no_grad():
                # Tensor tadi dimasukkan ke forward pass dr model untuk melakukan prediction
                outputs = self.model(input_ids, attention_mask)
                intent_pred = torch.argmax(outputs["intent_logits"], dim=1).item() #Hasil inference intent
                slot_preds = outputs["slot_preds"][0] #Hasil inference untuk NER

            # Convert hasil prediksi dari id ke format yang dapat dibaca manusia 
            intent_label = id2intent[intent_pred]
            # Menentukan confidence dari hasil prediksi 
            intent_probs = F.softmax(outputs["intent_logits"], dim=1)
            intent_conf = intent_probs[0][intent_pred].item()
            # Convert hasil prediksi NER dari id ke format yang bisa dibaca
            slot_labels = [id2slot[slot] for slot in slot_preds]
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

            # Print log hasil inference
            logger.info(f"Intent prediction: {intent_label} (confidence: {intent_conf:.4f})")

            # Ambil teks yang sudah ditokenisasi berserta label hasil prediksi NER tiap token
            filtered_tokens = []
            filtered_slots = []
            # Filter out semua special token dan hanya include token yang defined di model (id2slot)
            for token, slot in zip(tokens, slot_labels):
                if token not in self.tokenizer.all_special_tokens:
                    filtered_tokens.append(token)
                    filtered_slots.append(slot)

            # Parsing hasil NER untuk mapping qty dan menu - now returns invalid menus too
            parsed_items, invalid_menus = self.parse_slots(filtered_tokens, filtered_slots)
            logger.info(f"Parsed items: {parsed_items}")
            logger.info(f"Invalid menus: {invalid_menus}")

            return {
                "intent": intent_label,
                "intent_confidence": intent_conf,
                "tokens": filtered_tokens,
                "slots": filtered_slots,
                "parsed": parsed_items,
                "invalid_menus": invalid_menus  # Add invalid menus to return
            }
        except Exception as e:
            logger.error(f"NLP processing error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
        
class WhisperASRProcessor:
    """
    Class untuk loading model ASR Whisper & Proses transkripsi
    """
    #Function initialize model whisper large v3 turbo dengan  set languange indonesia
    def __init__(self, language="id", model_id="openai/whisper-large-v3-turbo"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu" # Select gpu by default jika tersedia
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32 # Use half precision f16 untuk optimize memory usage di gpu
        self.model_id = model_id
        self.language = language
        self.sample_rate = 16000 # Expected sample rate input oleh Whisper by default
        
        logger.info(f"Initializing Whisper model on {self.device} with dtype {self.torch_dtype}")
        
        # Load model menggunakan library huggingface untuk otomatis set weight dan penambahan 
        # flash_attention untuk optimisasi dalam processing agar lebih cepat
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
        # Otomatis load tokenizer dan parameter lainnya spesifik untuk model dan bahasa yang digunakan
        self.processor = AutoProcessor.from_pretrained(self.model_id, language=language)
        
        # Create pipeline otomatis dengan library huggingface juga
        self.pipe = pipeline(
            "automatic-speech-recognition", # task yang akan dilakukan
            model=self.model, # model yang diload sblmnya
            # Komponen dari processor yang diload tadi sebelumnya
            tokenizer=self.processor.tokenizer, 
            feature_extractor=self.processor.feature_extractor,
            chunk_length_s=20, # process audio dalam chunk per 20s (default standard library HF)
            batch_size=16, # Process dapat dilakukan 16 chunks secara bersamaan (default library HF)
            torch_dtype=self.torch_dtype,
            generate_kwargs={
                "language": language,
                "task": "transcribe"
            },
        )
        
        logger.info(f"ASR Processor initialized for language: {language}")
    
    def transcribe_audio(self, audio_array):
        """
        Melakukan transkripsi audio array dari client
        """
        try:
            # Menghitung durasi (T = N/Fs)
            duration = len(audio_array) / self.sample_rate
            logger.info(f"Transcribing audio chunk of length {duration:.2f}s")
            
            # Input array audio ke dalam pipeline sblmnya yangt elah dibuat dan output berupa teks transkripsi
            result = self.pipe(
                {"raw": audio_array, "sampling_rate": self.sample_rate},
                generate_kwargs={"language": self.language}
            )
            
            return result["text"]
        except Exception as e:
            logger.error(f"ASR transcription error: {e}")
            return None


class AudioServer:
    """
    Server class untuk melakukan handling koneksi WebSocket/UDP dan audio streaming serta forwarding ke bagian inference
    """
    def __init__(self, asr_processor, nlp_processor, websocket_port=8765, udp_port=9876):
        self.asr_processor = asr_processor # Simpan reference ke class processor untuk ASR
        self.nlp_processor = nlp_processor # Simpan reference ke class processor untuk NLP
        self.websocket_port = websocket_port # Port websocket yg dibuka server 
        self.udp_port = udp_port # Port UDP untuk discovery
        
        # Intent confidence threshold
        self.intent_confidence_threshold = 0.75
        
        # Audio buffering
        self.audio_buffer = deque()
        self.buffer_lock = Lock()
        self.is_listening = False
        
        logger.info(f"Audio Server initialized on ports WS:{websocket_port}, UDP:{udp_port}")

    # Function untuk handle & cek koneksi antara server dan client
    async def handle_connection(self, websocket):
        logger.info("Client connected")
        
        try:
            async for message in websocket:
                if isinstance(message, str):
                    await self._handle_text_message(websocket, message)
                else:
                    await self._handle_audio_message(message)
        except websockets.exceptions.ConnectionClosedError:
            logger.info("Client disconnected abruptly")
        except Exception as e:
            logger.error(f"Connection error: {e}")
        finally:
            logger.info("Client disconnected")

    # Function untuk menerima pesan dari client request untuk proses inference
    async def _handle_text_message(self, websocket, message):
        """Handle text-based control messages"""
        try:
            data = json.loads(message)
            
            # Jika message dr client start_listen dia mulai menyimpan audio chunk yang dikirim dari client ke dlm buffer
            if data['type'] == 'start_listen':
                logger.info(f"Starting transcription")
                self.is_listening = True
                self.audio_buffer.clear()
                await websocket.send(json.dumps({"status": "listening_started"}))

            # Jika message dr client stop_listen maka akan audio dalam buffer akan diproses 
            elif data['type'] == 'stop_listen':
                logger.info("Stopping transcription...")
                self.is_listening = False
                await self._process_buffered_audio(websocket)
                
        except json.JSONDecodeError:
            await websocket.send(json.dumps({
                "status": "error",
                "message": "Invalid JSON format"
            }))
        except Exception as e:
            await websocket.send(json.dumps({
                "status": "error",
                "message": f"Message processing error: {str(e)}"
            }))

    async def _handle_audio_message(self, message):
        """Handle binary audio data"""
        if self.is_listening:
            audio_chunk = np.frombuffer(message, dtype=np.float32)
            with self.buffer_lock:
                self.audio_buffer.extend(audio_chunk)
                logger.debug(f"Received audio chunk, buffer now has {len(self.audio_buffer)/16000:.2f}s")

    async def _process_buffered_audio(self, websocket):
        """Process the buffered audio through ASR and NLP"""
        with self.buffer_lock:
            if len(self.audio_buffer) == 0:
                await websocket.send(json.dumps({
                    "status": "error",
                    "message": "No audio received in buffer"
                }))
                return

            audio_array = np.array(self.audio_buffer)
            
            # Memasukkan audio array ke dalam processor ASR
            transcription = self.asr_processor.transcribe_audio(audio_array)
            if not transcription:
                await websocket.send(json.dumps({
                    "status": "error",
                    "message": "ASR transcription failed"
                }))
                return

            # Membersihkan output transkripsi
            clean_transcription = normalize_numbers_in_transcription(transcription)
            logger.info(f"Clean Transcription: {clean_transcription}")

            
            # Memasukkan hasil transkripsi yang sudah dibersihkan ke processor NLP
            nlp_result = self.nlp_processor.process_text(clean_transcription)
            if nlp_result:
                response = self._build_success_response(
                    clean_transcription, nlp_result
                )
            else:
                response = {
                    "status": "error",
                    "message": "Failed to process transcription with NLP",
                    "transcription": clean_transcription
                }

            await websocket.send(json.dumps(response))
            self.audio_buffer.clear()

    # function untuk membuat message response yang akan dikirimkan ke client kembali jika berhasil
    def _build_success_response(self, transcription, nlp_result):
        """Build successful response"""
        intent = nlp_result["intent"]
        intent_confidence = float(nlp_result["intent_confidence"])
        
        # Kalau intent dibawah treshold confidence anggap invalid out of scope
        if intent_confidence < self.intent_confidence_threshold:
            logger.info(f"Intent confidence {intent_confidence:.4f} below threshold {self.intent_confidence_threshold}, setting to 'oos'")
            response = {
                "status": "success",
                "transcription": transcription,
                "intent": "oos",
                "intent_confidence": intent_confidence,
                "parsed_order": [],  # Empty parsed order for low confidence
            }
        else:
            response = {
                "status": "success",
                "transcription": transcription,
                "intent": intent,
                "intent_confidence": intent_confidence,
                "parsed_order": [
                    {"menu": menu, "qty": qty} 
                    for qty, menu in nlp_result["parsed"]
                ],
            }
            
            # Tambahkan field invalid menus jika ada menu diluar list menu yg terdaftar terdeteksi
            if nlp_result["invalid_menus"]:
                response["invalid_menu"] = nlp_result["invalid_menus"]
                logger.info(f"Adding invalid_menu to response: {nlp_result['invalid_menus']}")
        
        return response

    # Menjalankan UDP Discovery agar client bisa mencari IP dan port server
    async def run_udp_discovery_server(self):
        """Run UDP discovery server"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('', self.udp_port)) # Bind UDP server ke semua interface di port yang ditentukan
        sock.setblocking(False)
        
        logger.info(f"UDP discovery server listening on port {self.udp_port}")
        # Loop untuk menunggu discovery request yang dibroadcast dari client
        while True:
            try:
                await asyncio.sleep(0.1)
                try:
                    data, addr = sock.recvfrom(1024)
                    if data == b'where-are-you':
                        logger.info(f"Received discovery request from {addr}")
                        response = f"I-am-here:{self.websocket_port}".encode()
                        sock.sendto(response, addr)
                except BlockingIOError:
                    pass
            except Exception as e:
                logger.error(f"UDP discovery server error: {e}")
                await asyncio.sleep(0.5)

    # Function untuk menjalankan Websocket dan UDP server
    async def start_server(self):
        """Start both WebSocket and UDP servers"""
        udp_task = asyncio.create_task(self.run_udp_discovery_server())
        
        websocket_server = await websockets.serve(
            self.handle_connection, 
            "0.0.0.0", 
            self.websocket_port,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=10
        )
        
        logger.info(f"Audio Server started on ws://0.0.0.0:{self.websocket_port}")
        logger.info(f"UDP discovery server running on port {self.udp_port}")
        
        await asyncio.gather(
            websocket_server.wait_closed(),
            udp_task
        )

# Main FUnction
async def main():
    # Initialize processors
    asr_processor = WhisperASRProcessor(language="id")
    nlp_processor = NLPProcessor()  # Your existing NLP processor
    
    # Initialize server with processors
    server = AudioServer(
        asr_processor=asr_processor,
        nlp_processor=nlp_processor,
        websocket_port=8765,
        udp_port=9876
    )
    
    # Start server
    await server.start_server()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")