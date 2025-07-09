# Modul class voice client untuk handle status, state dan interaksi sistem
import json
import yaml
import asyncio
import time
import psycopg2
import modules.shared as shared
from modules.shared import ClientState, play_sound, send_to_app, NUMBER_MAPPING, TOF_STABLE_DETECTION_TIME, TOF_STABLE_UNDETECTION_TIME
    
class VoiceClient:
    def __init__(self):
        self.config = self.load_config()
        self.current_state = ClientState.WAITING_FOR_CAR #initialize first state waiting for car
        self.current_orders = {} #list untuk tracking pesanan saat ini
        self.car_detected = False #flag deteksi mobil
        self.last_car_detection_time = 0 #timestamp terakhir kali mobil kdetek
        self.last_car_undetection_time = 0 #timestamp terakhir kali mobil ud ga ada
        self.abort_recording = False #flag untuk berhentikan record jika case mobil ga kedeteksi
        self.ffmpeg_proc = None #reference ke ffmpeg subprocess untuk record
        self.car_status_changed = asyncio.Event()  # object asyncio untuk notifikasi perubahan status mobil ad/nggak
        self.recording_in_progress = False #flag untuk indikasi apakah recording active/nggak
        self.last_distance = 0  # Store last distance reading jarak mobil
        
        # Timing metrics
        self.order_start_time = None  # Full order timer
        self.intent_start_time = None  # order per intent processing timer
        self.intent_recording_start_time = None  # When recording actually starts
        self.intent_recording_end_time = None  # When recording ends
        self.intent_process_start_time = None  # When server processing starts
        self.intent_timings = []  # List to store each intent processing time with recording and processing times
        
    def load_config(self):
        """Load db config"""
        try:
            with open('/home/robs/Skripsi/drivethru/backend/modules/dbconfig.yaml', 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            print("❌ dbconfig.yaml not found!")
            return None
        except yaml.YAMLError as e:
            print(f"❌ Error parsing dbconfig.yaml: {e}")
            return None
    
    async def enter_waiting_state(self):
        """Function untuk reset client ke state awal yaitu waiting for car saat mobil pergi, clear list order & reset flag car detected"""
        self.current_state = ClientState.WAITING_FOR_CAR
        self.current_orders = {}
        self.car_detected = False

    async def abort_current_recording(self):
        """Function untuk berhentikan segala process recording"""
        #update flag abort record ke true
        self.abort_recording = True
        # jika subprocess ffmpeg masih running akan dikill
        if self.ffmpeg_proc:
            try:
                self.ffmpeg_proc.kill()
            except:
                pass
            finally:
                # cleanup process
                self.ffmpeg_proc = None
        #update balik flagnya ke false
        self.abort_recording = False
    
    async def handle_confirmed_state(self):
        """Function untuk menghandle jika kondisi state konfirmasi"""
        await self.abort_current_recording() # tutup mikrofon 
        #Kirim order ke database
        print("\n📦 Sending order to database:")
        print(json.dumps(self.current_orders, indent=2))
        
        # Load config database
        config = self.load_config()
        if not config:
            print("❌ Failed to load database config")
            return
        
        db_config = config['database']
        try:
            # Connect ke DB
            conn = psycopg2.connect(
                host= shared.ip_addr,
                database=db_config['name'],
                user=db_config['user'],
                password=db_config['password'],
                port=db_config['port']
            )
            cursor = conn.cursor()
            
            # Query harga per menu terus kalkulasi total harga
            total_price = 0
            for menu_name, quantity in self.current_orders.items():
                cursor.execute("SELECT price FROM menu WHERE name = %s", (menu_name,))
                result = cursor.fetchone()
                if result:
                    price = result[0]
                    total_price += price * quantity
                    print(f"{menu_name}: {quantity} x {price} = {price * quantity}")
                else:
                    print(f"Warning: Menu item '{menu_name}' not found in database!")
            
            print(f"Total Price: {total_price}")
            
            # Save transaksi ke DB
            cursor.execute("""
                INSERT INTO orders (menu_items, total_price, payment_status) 
                VALUES (%s, %s, %s) RETURNING id
            """, (json.dumps(self.current_orders), total_price, 'unpaid'))
            
            order_id = cursor.fetchone()[0]
            conn.commit()
            
            print(f"✅ Order saved to database with ID: {order_id}")
            
        except Exception as e:
            print(f"❌ Database error: {e}")
            if 'conn' in locals():
                conn.rollback()
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()
            #Clear order list
        self.current_orders = {}
        print("✅ Order sent to database!")
        #update state ke confirmed
        self.current_state = ClientState.CONFIRMED_STATE
    
        # Stop timer full order dan print summary kecepatan order & processing time
        if self.order_start_time is not None:
            full_order_time = time.time() - self.order_start_time
            print("\n==== ORDER TIMING SUMMARY ====")
            print(f"Summarized full order time = {full_order_time:.1f}s")
            
            if self.intent_timings:
                print("Processing time per intent:")
                for idx, (intent, record_time, process_time) in enumerate(self.intent_timings, 1):
                    print(f"{idx}. {intent} | record time:{record_time:.1f}s | process time:{process_time:.1f}s |")
            print("===============================")
        # Kasih tahu app kalau order udh confirmed untuk update UInya
        await send_to_app({
            "type": "order_confirmed",
            "message": "Pesanan selesai! Silahkan maju kedepan untuk melanjutkan"
        })
        # play sound konfirmasi pesanan selesai
        await play_sound("done")

    async def handle_canceled_state(self):
        """Function untuk menghandle jika kondisi state cancel order"""
        await self.abort_current_recording() #tutup mic stop record
        print("\n❌ Order canceled by user")
        # clear order list tracking
        self.current_orders = {}
        self.current_state = ClientState.CANCELED_STATE
        
        # Stop the full order timer and print summary when order is canceled
        if self.order_start_time is not None:
            full_order_time = time.time() - self.order_start_time
            print("\n==== ORDER TIMING SUMMARY ====")
            print(f"Summarized full order time = {full_order_time:.1f}s")
            
            if self.intent_timings:
                print("Processing time per intent:")
                for idx, (intent, record_time, process_time) in enumerate(self.intent_timings, 1):
                    print(f"{idx}. {intent} | record time:{record_time:.1f}s | process time:{process_time:.1f}s |")
            print("===============================")
        # Kasih tahu app kalau order dicancel dan dia update UI dari message ini
        await send_to_app({
            "type": "order_canceled",
            "message": "Pesanan dibatalkan. Semoga kita bisa bertemu lagi lain waktu!"
        })
        # Play audio pesanan dibatalkan
        await play_sound("cancel")

    # function untuk tracking order internal di client
    async def update_orders(self, items, is_add=True):
        """Update & tracking pesanan saat ini"""
        for menu, qty in items.items():
            # Convert qty ke int karena masih string dr NLP
            if isinstance(qty, str):
                try:
                    qty = int(qty)
                except ValueError:
                    qty_lower = qty.lower()
                    if qty_lower in NUMBER_MAPPING:
                        qty = NUMBER_MAPPING[qty_lower]
                    else:
                        # Default 1 kalau hanya menu saja tanpa qty
                        print(f"Warning: Could not parse quantity '{qty}' for {menu}, using 1")
                        qty = 1
            
            # Kalau add order ya tambah ke list
            if is_add:
                if menu in self.current_orders:
                    self.current_orders[menu] += qty
                else:
                    self.current_orders[menu] = qty
            # Kalau selain add order hanya ada remove order, kurangin dari list
            else:
                if menu in self.current_orders:
                    self.current_orders[menu] -= qty
                    # Remove item completely if quantity reaches 0 or less
                    if self.current_orders[menu] <= 0:
                        del self.current_orders[menu]

    async def handle_car_detection(self, detected):
        """Handle deteksi mobil & pergantian state"""
        current_time = time.time() # catet timestamp saat ini buat itung durasi berapa lama mobil deteksi/tidak terdeteksi.
        status_changed = False
        
        # Jika detected true dari pembacaan ToF
        if detected == True:
            if not self.car_detected:
                # update flag car detected dan status change
                self.last_car_detection_time = current_time
                self.car_detected = True
                self.abort_recording = False
                status_changed = True
                if hasattr(self, 'last_car_undetection_time'):
                    del self.last_car_undetection_time
                
                # Start timer untuk track order time begitu car detect
                self.order_start_time = current_time
                self.intent_timings = []  # Clear timing sebelumnya untuk intent processing
                
                # Transisi state ke car detected
                if self.current_state == ClientState.WAITING_FOR_CAR:
                    self.current_state = ClientState.CAR_DETECTED
                    # Kirim ke app status update car detect
                    await send_to_app({
                        "type": "car_detected",
                    })
                    # Play sound selamat datang
                    await play_sound("welcome")
                    # Transisi ke init_state
                    self.current_state = ClientState.INIT_STATE
                    
            elif (current_time - self.last_car_detection_time) >= TOF_STABLE_DETECTION_TIME:
                # Stable detection - no state change if already in process
                pass
                
        elif detected == False:  # Car not detected
            if self.car_detected:
                if not hasattr(self, 'last_car_undetection_time'):
                    self.last_car_undetection_time = current_time
                elif (current_time - self.last_car_undetection_time) >= TOF_STABLE_UNDETECTION_TIME:
                    # Mobil sudah tidak terdeteksi lebih lama dari undetection time yag sudah diset
                    await self.abort_current_recording() # abort segala process record
                    # Transisi kembali ke state waiting for car
                    self.current_state = ClientState.WAITING_FOR_CAR
                    # Clear list order & ganti flag status car detect & status change 
                    self.current_orders = {}
                    self.car_detected = False
                    status_changed = True
                    del self.last_car_undetection_time
                    
                    # Reset timing data
                    self.order_start_time = None
                    self.intent_timings = []

                    await send_to_app({
                        "type": "car_gone",
                    })
        
        # Set the event if the car status changed
        if status_changed:
            self.car_status_changed.set()
