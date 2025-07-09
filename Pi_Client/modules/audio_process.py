import audioop
import webrtcvad
import time
import subprocess
import numpy as np
import sys
import json
import asyncio
import websockets
import modules.shared as shared
from modules.shared import logger, send_to_app, play_sound, ClientState, NUMBER_MAPPING, MAX_RECORDING_DURATION, SILENCE_TIMEOUT, MIN_VOLUME, VAD_MODE, SAMPLERATE_IN,PULSE_DEVICE, CHUNK_SIZE, CHANNELS, CONSECUTIVE_VOICE

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
async def monitor_car_status(client):
    """Function untuk monitor status mobil ada/nggak"""
    try:
        while True:
            # Wait for car status to change
            await client.car_status_changed.wait()
            
            # Check if car is gone
            if not client.car_detected:
                # Force abort the recording
                client.abort_recording = True
                
            # Reset the event
            client.car_status_changed.clear()
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"Error in car status monitor: {e}")
        
def calculate_noise_floor(chunks):
    """Calculate noise floor from initial audio chunks"""
    rms_values = [audioop.rms(chunk, 2) for chunk in chunks if len(chunk) >= 2]
    return np.percentile(rms_values, 50) if rms_values else MIN_VOLUME

def is_voice(audio_chunk, vad, sample_rate, volume_threshold):
    """Cek apakah chunk ada aktivitas suara dengan treshold volume minimal & VAD untuk konfirmasi"""
    if len(audio_chunk) < 2:
        return False
    
    rms = audioop.rms(audio_chunk, 2)
    if rms < volume_threshold:
        return False
    
    try:
        return vad.is_speech(audio_chunk, sample_rate)
    except:
        return False

# =============================================================================
# MAIN RECORDING LOGIC
# =============================================================================
async def record_with_vad(client):
    """Recording function"""
    # Initialize VAD
    vad = webrtcvad.Vad()
    vad.set_mode(VAD_MODE)
    last_vad_reset = time.time()
    
    # Untuk kalibrasi noise floor digunakan 2 detik audio data 
    calibration_chunks = []
    calibration_samples = int(1.0 * SAMPLERATE_IN / CHUNK_SIZE * 2)
    
    # Command ffmpeg untuk record dijalankan dgn subprocess
    ffmpeg_cmd = [
        "ffmpeg",
        "-f", "pulse",
        "-i", PULSE_DEVICE,
        "-ar", str(SAMPLERATE_IN),
        "-ac", str(CHANNELS),
        "-f", "s16le",
        "-loglevel", "quiet",
        "-"
    ]

    audio_buffer = bytearray() # buffer untuk simpan recording
    is_recording = False # flag status recording
    last_voice_time = time.time() # timestamp last kedetek suara
    start_time = time.time() # start time
    consecutive_voice = 0
    last_status_update = 0
    dynamic_threshold = MIN_VOLUME
    
    # Memulai status recording ud dimulai
    client.recording_in_progress = True
    # Stat subprocess ffmpeg untuk record
    client.ffmpeg_proc = subprocess.Popen(
        ffmpeg_cmd,
        stdout=subprocess.PIPE,
        bufsize=CHUNK_SIZE
    )

    print("\nCalibrating noise floor...", end="\r")
    
    try:
        # Start create task anychronous untuk check mobil masih ada/gak in paralel
        car_status_monitor = asyncio.create_task(monitor_car_status(client))
        
        # Terus looping untuk menyimpan chunk hingga cukup untuk kalibrasi noise
        while len(calibration_chunks) < calibration_samples:
            # Kalau status mobil g kedetek/ abort record dia bakal cancel task monitor mobil td
            if not client.car_detected or client.abort_recording:
                car_status_monitor.cancel()
                return None
            
            # Baca chunk audio dari subprocess ffmpeg record td
            chunk = await asyncio.get_event_loop().run_in_executor(
                None, lambda: client.ffmpeg_proc.stdout.read(CHUNK_SIZE) if client.ffmpeg_proc else None
            )
            
            if not chunk:
                await asyncio.sleep(0.01)  # Small sleep to prevent CPU overload
                continue
            
            # input chunkny untuk dikalibrasi noise floornya
            calibration_chunks.append(chunk)
        
        # Melakukan calibration dan menentukan treshold suara yang dianggap sebagai voice
        if calibration_chunks and not client.abort_recording:
            # Treshold didapat dari hasil calculate noise floor dikalikan 1.5 (tidak akan bisa lebih rendah dari MIN_VOLUME)
            dynamic_threshold = max(MIN_VOLUME, calculate_noise_floor(calibration_chunks) * 1.5)
        
        # BEritahu app kalau untuk update UI memberitahu user udh bisa untuk mulai berbicara
        if client.current_state == ClientState.INIT_STATE or ClientState.PROCESS_STATE:
            await send_to_app({
                "type": "start_speak",
            })
        print(f"Listening... (threshold: {dynamic_threshold:.0f})", end="\r")
        # pre-buffer untuk simpan audio 1s sebelum voice kedeteksi untuk mencegah kata pertama terpotong
        pre_buffer = bytearray()
        pre_buffer_max_size = int(1.0 * SAMPLERATE_IN)  # 1 second of audio
        
        # Loop untuk recording utama
        while shared.is_running and not client.abort_recording:
            # Kalau mobil hilang lgsung stop record
            if not client.car_detected:
                print("\n🛑 Car left - aborting recording")
                break

            # Baca chunk audio
            try:
                chunk = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, lambda: client.ffmpeg_proc.stdout.read(CHUNK_SIZE) if client.ffmpeg_proc else None
                    ),
                    timeout=0.1  # 100ms timeout
                )
            except asyncio.TimeoutError:
                # No data ready, continue loop to check car status
                continue
                
            if not chunk or len(chunk) != CHUNK_SIZE or client.abort_recording:
                break

            current_time = time.time()
            elapsed = current_time - start_time
            
            # Reinitialize VAD tiap 5 detik biar lebih stabil & konsisten
            if current_time - last_vad_reset > 5.0:
                vad = webrtcvad.Vad()
                vad.set_mode(VAD_MODE)
                last_vad_reset = current_time
            # Jika status blm mulai recording
            if not is_recording:
                # Check apakah ada aktivitas suara 
                voice_detected = is_voice(chunk, vad, SAMPLERATE_IN, dynamic_threshold)
                
                if voice_detected:
                    consecutive_voice += 1
                    last_voice_time = current_time
                    
                    # Ambil pre buffer dari voice sblm terdeteksi
                    pre_buffer.extend(chunk)
                    if len(pre_buffer) > pre_buffer_max_size:
                        pre_buffer = pre_buffer[-pre_buffer_max_size:]
                    
                    # Start recording jika voice detected lebih banyak dari consecutive voice
                    if consecutive_voice >= CONSECUTIVE_VOICE:
                        is_recording = True
                        recording_start = current_time
                        # Start timer order time
                        client.intent_start_time = current_time
                        client.intent_recording_start_time = current_time
                        
                        # Tambahkan prebuffer ke recording 
                        audio_buffer.extend(pre_buffer)
                        audio_buffer.extend(chunk)
                        
                        print(f"\n✅ Recording started! (Voice confirmed at {elapsed:.1f}s)")
                        await send_to_app({"type": "start_listening"})
                else:
                    # Kalau gak ada suara bakal terus scan treshold noise nunggu aktiviytas suara
                    consecutive_voice = 0
                    # Keep some silence in pre-buffer but limit its size
                    pre_buffer.extend(chunk)
                    if len(pre_buffer) > pre_buffer_max_size:
                        pre_buffer = pre_buffer[-pre_buffer_max_size//2:]
                    rms = audioop.rms(chunk, 2)
                    dynamic_threshold = max(MIN_VOLUME, rms * 1.3)
                    
            # Selama recording masih aktif 
            else:
                audio_buffer.extend(chunk)
                
                rms = audioop.rms(chunk, 2) if len(chunk) >= 2 else 0
                if rms >= dynamic_threshold:
                    last_voice_time = current_time
                elif (current_time - last_voice_time) > SILENCE_TIMEOUT:
                    actual_duration = current_time - recording_start
                    print(f"\n🔇 Silence detected. Stopping recording ({actual_duration:.1f}s)")
                    # Set recording end time - NEW
                    client.intent_recording_end_time = current_time
                    break
                
                if (current_time - recording_start) > MAX_RECORDING_DURATION:
                    print(f"\n⚠️ Maximum recording time reached ({MAX_RECORDING_DURATION}s)")
                    # Set recording end time on max duration too - NEW
                    client.intent_recording_end_time = current_time
                    break

            if current_time - last_status_update > 0.1 or not is_recording:
                status = "🔴 Listening" if not is_recording else f"🟢 Recording ({len(audio_buffer)/SAMPLERATE_IN/2:.1f}s)"
                # Include distance in status display during recording
                if hasattr(client, 'last_distance'):
                    distance_info = f" | Distance: {client.last_distance}cm"
                else:
                    distance_info = ""
                sys.stdout.write(f"\rElapsed: {elapsed:.1f}s | {status} | Thresh: {dynamic_threshold:.0f}{distance_info}")
                sys.stdout.flush()
                last_status_update = current_time

    except Exception as e:
        logger.error(f"Error in audio processing: {e}")
    finally:
        # Stop task async tadi untuk monitor keadaan mobil saat recording selesai
        if car_status_monitor and not car_status_monitor.done():
            car_status_monitor.cancel()
        # Stop subprocess ffmpeg
        if client.ffmpeg_proc:
            try:
                client.ffmpeg_proc.kill()
            except:
                pass
            client.ffmpeg_proc = None
        client.abort_recording = False
        client.recording_in_progress = False

    # Return audio data dr buffer dalam bentuk array numpy
    if len(audio_buffer) > 0 and not client.abort_recording:
        audio_data = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
        return audio_data
    return None

# =============================================================================
# MAIN WORKFLOW CONTROLLER
# =============================================================================
async def continuous_recording(client):
    """Processing audio serta workflow control antar server & aplikasi"""
    while shared.is_running:
        # hanya record jika client dalam state init/process
        if client.current_state not in [ClientState.INIT_STATE, ClientState.PROCESS_STATE]:
            await asyncio.sleep(0.1)
            continue
            
        # Cek apakah mobil masih terdeteksi/nggak
        if not client.car_detected:
            if client.current_state != ClientState.WAITING_FOR_CAR:
                await client.enter_waiting_state()
            continue

        print("\nStarting recording session...")  # Debug print
        # panggil function record with vad untuk mulai sesi record
        audio = await record_with_vad(client)
        if audio is None:
            continue
        
        # Jika berhasil return audio maka akan kirim ke server
        try:
            # Double-check state before processing
            if client.current_state not in [ClientState.INIT_STATE, ClientState.PROCESS_STATE]:
                continue

            # Start timer untuk server process time
            client.intent_process_start_time = time.time()

            # Kirim ke server start listen biar tau kalau audio bakal masuk via websocket
            async with websockets.connect(shared.inference_addr, ping_timeout=20) as inference_ws:
                await inference_ws.send(json.dumps({
                    "type": "start_listen", 
                }))
                await inference_ws.recv() # Wait response server

                # Send audio dalam chunk-chunk
                chunk_size = 16000
                for i in range(0, len(audio), chunk_size):
                    # Kalau mobil hilang lgsg abort juga
                    if client.abort_recording or not client.car_detected:
                        print("\n🛑 Aborting audio processing - car moved away")  # Debug print
                        break
                    chunk = audio[i:i + chunk_size]
                    await inference_ws.send(chunk.tobytes())
                # Kirim pesan ke server kalau audio ud dikirim semua boleh mulai proses inference
                await inference_ws.send(json.dumps({"type": "stop_listen"}))
                # Kirim pesan ke app untuk update status UI kasih tau user lg proses
                await send_to_app({
                    "type": "processing_order",
                    "message": "Memproses pesanan anda..."
                })
                # Tunggu reply dr server
                result = await inference_ws.recv()
                result = json.loads(result)
                
                # End of processing time - NEW
                process_end_time = time.time()
                
                # Print the received inference result
                print("\n🔍 Received from inference server:")
                print(f"  Transcription: \"{result.get('transcription', 'N/A')}\"")
                print(f"  Intent: {result.get('intent', 'N/A')}")
                print(f"  Parsed order: {json.dumps(result.get('parsed_order', []), indent=2)}")
                if result.get('invalid_menu'):
                    print(f"  Invalid menus: {result.get('invalid_menu')}")
                print("-------------------------------------------")
                
            # Simpan processing time untuk data
            if (client.intent_recording_start_time is not None and 
                client.intent_recording_end_time is not None and
                client.intent_process_start_time is not None):
                
                # Calculate separate times
                record_duration = client.intent_recording_end_time - client.intent_recording_start_time
                process_duration = process_end_time - client.intent_process_start_time
                
                if result.get("status") == "success":
                    intent = result["intent"]
                    # Add the intent with both record and process time
                    client.intent_timings.append((intent, record_duration, process_duration))
                    
                # Reset all timing variables for next cycle
                client.intent_start_time = None
                client.intent_recording_start_time = None
                client.intent_recording_end_time = None
                client.intent_process_start_time = None

            # Jika reply dr server success mulai proses dan kirim ke apps untuk update
            if result.get("status") == "success":
                intent = result["intent"]
                parsed_items = result["parsed_order"]
                
                # Convert qty ke integer sblm dikirim ke apps
                for item in parsed_items:
                    if "qty" in item and isinstance(item["qty"], str):
                        try:
                            item["qty"] = int(item["qty"])
                        except ValueError:
                            qty_lower = item["qty"].lower()
                            if qty_lower in NUMBER_MAPPING:
                                item["qty"] = NUMBER_MAPPING[qty_lower]
                            else:
                                # Default to 1 if we can't parse
                                print(f"Warning: Could not parse quantity '{item['qty']}' for {item['menu']}, using 1")
                                item["qty"] = 1
                                
                invalid_menus = result.get("invalid_menu", [])

                # Jika masih di INIT_STATE maka hanya bisa menerima add_item dan cancel_order intent
                INIT_STATE_ALLOWED_INTENTS = {'add_item', 'cancel_order'}

                if client.current_state == ClientState.INIT_STATE and intent not in INIT_STATE_ALLOWED_INTENTS:
                    await send_to_app({
                        "type": "no_intent",
                        "message": "Tidak dapat mengurangi/mengganti pesanan diawal mohon lihat panduan kembali"
                    })
                    await play_sound("invalid_intent")
                    continue
            
                # Kalau cancel dan confirm intent ada perlu handle state function sendiri karena ada perubahan state
                if intent == 'cancel_order':
                    await client.handle_canceled_state()
                    continue
                elif intent == 'confirm_order':
                    await client.handle_confirmed_state()
                    continue
                elif intent == "oos":
                    await send_to_app({
                        "type": "no_intent",
                        "message": "Maaf intensi anda kurang jelas mohon berbicara kembali"
                    })
                    await play_sound("invalid_intent")
                    
                # Untuk 3 intent pemesanan utama
                elif intent == "add_item":
                    if client.current_state == ClientState.INIT_STATE:
                        client.current_state = ClientState.PROCESS_STATE
                    items = {item["menu"]: item["qty"] for item in parsed_items}
                    
                    # Update tracking order list
                    await client.update_orders(items, is_add=True)
                    # Kasih tahu app untuk nambah item ke list
                    # Kalau ada menu yg gk ad di daftar menu & masih ada yg valid
                    if invalid_menus and items:
                        await send_to_app({
                            "type": "add_item",
                            "items": items,
                            "message": f"Ada beberapa menu yang tidak ditambahkan karena tidak terdaftar di menu kami: {', '.join(invalid_menus)}"
                        })
                        await play_sound("invalid_with_valid")
                    # Jika hanya kedeteksi menu yg gak ada di daftar menu
                    elif invalid_menus and not items:
                        await send_to_app({
                            "type": "invalid_menu",
                            "message": f"Tidak ada menu yang ditambahkan karena tidak terdaftar di menu kami: {', '.join(invalid_menus)}"
                        })
                        await play_sound("invalid")
                    # Jika valid hanya kedetek menu yg ada di daftar menu
                    elif items:  
                        await send_to_app({
                            "type": "add_item",
                            "items": items,
                            "message": "Pesanan diperbarui, mohon cek kembali pesanan anda"
                        })
                        await play_sound("valid")
                    
                elif intent == "remove_item":
                    items = {item["menu"]: item["qty"] for item in parsed_items}
                    
                    # Kita cek dlu apakah item yang akan diremove itu ada gak di pesanan saat ini
                    items_not_in_order = []
                    valid_removal_items = {}
                    
                    for menu, qty in items.items():
                        if menu not in client.current_orders:
                            items_not_in_order.append(menu)
                        else:
                            # Check if requested quantity is available
                            available_qty = client.current_orders[menu]
                            if qty > available_qty:
                                # If they want to remove more than available, remove all available
                                valid_removal_items[menu] = available_qty
                                print(f"Warning: Requested to remove {qty} {menu}, but only {available_qty} available. Removing all.")
                            else:
                                valid_removal_items[menu] = qty
                    
                    # Update list order terlebih dahulu kalau ada pengurangan yang valid bisa dilakuin
                    if valid_removal_items:
                        await client.update_orders(valid_removal_items, is_add=False)
                    
                    # Jika menu tidak ada di daftar menu & belum ditambahin ke list order saat ini
                    if invalid_menus and items_not_in_order:
                        combined_errors = invalid_menus + items_not_in_order
                        await send_to_app({
                            "type": "remove_item" if valid_removal_items else "item_not_in_order",
                            "items": valid_removal_items if valid_removal_items else {},
                            "message": f"Ada menu yang tidak dikurangi: {', '.join(combined_errors)} (tidak terdaftar di menu atau belum ada dalam pesanan)"
                        })
                        await play_sound("invalid_with_valid" if valid_removal_items else "item_not_in_order")
                    elif invalid_menus:
                        # Only invalid menus
                        await send_to_app({
                            "type": "remove_item" if valid_removal_items else "invalid_menu",
                            "items": valid_removal_items if valid_removal_items else {},
                            "message": f"Ada menu yang tidak dikurangi karena tidak terdaftar di menu kami: {', '.join(invalid_menus)}"
                        })
                        await play_sound("invalid_with_valid" if valid_removal_items else "invalid")
                    elif items_not_in_order:
                        # Only items not in order
                        await send_to_app({
                            "type": "remove_item" if valid_removal_items else "item_not_in_order",
                            "items": valid_removal_items if valid_removal_items else {},
                            "message": f"Ada menu yang tidak dikurangi karena belum ada dalam pesanan Anda: {', '.join(items_not_in_order)}"
                        })
                        await play_sound("not_in_order_with_valid" if valid_removal_items else "item_not_in_order")
                    elif valid_removal_items:
                        # Only successful removals
                        await send_to_app({
                            "type": "remove_item",
                            "items": valid_removal_items,
                            "message": "Pesanan diperbarui, mohon cek kembali pesanan anda"
                        })
                        await play_sound("valid")
                        
                elif intent == "change_item":
                    if len(parsed_items) >= 2:
                        # For change_item, check if the first item (to be removed) exists in current order
                        remove_menu = parsed_items[0]["menu"]
                        remove_qty = parsed_items[0]["qty"]
                        add_menu = parsed_items[-1]["menu"]
                        add_qty = parsed_items[-1]["qty"]
                        
                        # Check if the item to be changed exists in current order
                        if remove_menu not in client.current_orders:
                            await send_to_app({
                                "type": "item_not_in_order",
                                "message": f"Maaf, {remove_menu} belum ada dalam pesanan Anda sehingga tidak bisa diganti"
                            })
                            await play_sound("item_not_in_order")
                            continue
                        
                        # Check if requested quantity to change is available
                        available_qty = client.current_orders[remove_menu]
                        if remove_qty > available_qty:
                            await send_to_app({
                                "type": "insufficient_quantity",
                                "message": f"Maaf, hanya ada {available_qty} {remove_menu} dalam pesanan, tidak bisa mengganti sejumlah {remove_qty}",
                            })
                            await play_sound("insufficient_quantity")
                            continue
                        
                        # If validation passes, proceed with the change
                        remove_item = {remove_menu: remove_qty}
                        add_item = {add_menu: add_qty}
                        
                        # Update tracking order list & Play audio konfirmasi
                        await client.update_orders(remove_item, is_add=False)
                        await client.update_orders(add_item, is_add=True)
                  
                        await send_to_app({
                            "type": "remove_item",
                            "items": remove_item
                        })
                        await send_to_app({
                            "type": "add_item",
                            "items": add_item
                        })
                        await play_sound("valid")
                        
                    elif invalid_menus:
                        await send_to_app({
                            "type": "invalid_menu",
                            "message": f"Penggantian pesanan gagal, menu berikut tidak terdaftar di menu kami: {', '.join(invalid_menus)}",
                        })
                        await play_sound("invalid")
                        
                    else:
                        # Handle incomplete change_item intent
                        await send_to_app({
                            "type": "incomplete_change",
                            "message": "Penggantian pesanan gagal, untuk mengganti pesanan mohon sebutkan menu yang ingin diganti dan menu penggantinya"
                        })
                        await play_sound("incomplete_change")

                # Print update tracking order saat ini jika mengalami perubahan dr 3 intent tsb
                if intent in ("add_item", "remove_item", "change_item"):
                    print(f"\nCurrent order: {json.dumps(client.current_orders, indent=2)}")
                    
            elif result.get("status") == "error":
                error_msg = result.get("message", "Unknown error")
                print(f"\n❌ Error: {error_msg}")
                await send_to_app({
                    "type": "error",
                    "message": error_msg
                })
                # Reset all timing variables on errors
                client.intent_start_time = None
                client.intent_recording_start_time = None
                client.intent_recording_end_time = None
                client.intent_process_start_time = None

        except websockets.exceptions.ConnectionClosedError:
            print("\n⚠️ Connection to inference server closed")
            await send_to_app({
                "type": "inference_disconnect",
                "message": "Connection to server lost"
            })
            # Reset semua timer kalau masalah koneksi
            client.intent_start_time = None
            client.intent_recording_start_time = None
            client.intent_recording_end_time = None
            client.intent_process_start_time = None
            await asyncio.sleep(1)
        
        except Exception as e:
            print(f"\n⚠️ Recording error: {str(e)}")  # Debug print
            await send_to_app({
                "type": "error",
                "message": f"Recording error: {str(e)}"
            })
            # Reset semua timer kalau ad exception
            client.intent_start_time = None
            client.intent_recording_start_time = None
            client.intent_recording_end_time = None
            client.intent_process_start_time = None
            await asyncio.sleep(1)