import busio
import board
import asyncio
import adafruit_vl53l1x
import modules.shared as shared
from modules.shared import logger, TOF_DETECTION_THRESHOLD

async def monitor_tof_sensor(client):
    """Function reading TOF sensor terus2an"""
    # initialize I2C & sensor VL53L1X
    i2c = busio.I2C(board.SCL, board.SDA)
    vl53 = adafruit_vl53l1x.VL53L1X(i2c)
    # set mode distance 2 untuk read long range up to 3m
    vl53.distance_mode = 2  
    vl53.timing_budget = 200  
    vl53.start_ranging()
    
    print("VL53L1X Ready. Monitoring distance...")
    
    try:
        while shared.is_running:
            if vl53.data_ready:
                distance = vl53.distance
                vl53.clear_interrupt()
                
                # Jika reading tidak terbaca oleh sensor karena kejauhan anggap 3m
                if distance is None:
                    distance = 300  
                    print(f"⚠️ No valid reading, using fallback: {distance}cm")
                
                # Mengambil info state client saat ini
                current_state = client.current_state
                
                # Cek Mobil ada didalam jarak deteksi 50cm atau tidak
                if distance < TOF_DETECTION_THRESHOLD:
                    car_detected = True
                else:
                    car_detected = False
                
                # Simpen jarak terakhir terdeteksi untuk di print dan cek mobil ada/nggak
                client.last_distance = distance
                
                # Print state client & jarak kendaraan saat ini hanya saat tidak sedang membuka recording
                if not hasattr(client, 'recording_in_progress') or not client.recording_in_progress:
                    print(f"Distance: {distance}cm | State: {current_state}", end="\r")
                
                # Panggil method handle_car_detection dari VoiceClient untuk update dan pergantian state sistem di awal
                await client.handle_car_detection(car_detected)
                
            await asyncio.sleep(0.05)  
    except Exception as e:
        logger.error(f"TOF sensor error: {e}")
    finally:
        vl53.stop_ranging()