# Module untuk handle koneksi websocket apps, server dan discovery ip server juga
import asyncio
import websockets
import socket
import modules.shared as shared
from modules.voiceclient import VoiceClient
from modules.tof import monitor_tof_sensor
from modules.audio_process import continuous_recording
from modules.shared import logger, send_to_app, APP_PORT

# Function UDP Discovery broadcast untuk cari IP dan PORT server
async def udp_discovery(sock):
    while True:
        try:
            print("Sending discovery broadcast...")
            sock.sendto(b'where-are-you', ('<broadcast>', 9876))
            data, addr = sock.recvfrom(1024)

            if data.startswith(b'I-am-here'):
                decoded_data = data.decode()
                print(f"Server found at {addr[0]}, replied with: {decoded_data}")
                shared.ip_addr = addr[0]
                # Parse the port number from the response
                port = decoded_data.split(':', 1)[1] if ':' in decoded_data else '6969'
                
                # update inference_addr jadi websocket address
                shared.inference_addr = f"ws://{addr[0]}:{port}"
                print(f"Set inference address to: {shared.inference_addr}")
                return  
            else:
                print("Received unexpected reply. Trying again...")

        except socket.timeout:
            print("No server found (timeout). Retrying...")
        
        # Wait a bit before trying again
        await asyncio.sleep(1)

async def maintain_inference_connection():
    """Function untuk mengecek koneksi antara client dengan inference server"""
    shared.inference_connected = False # diawal statusnya false karena blm terhubung
    
    # Selama program masih berjalan (shared.is_running=true)
    while shared.is_running:
        try:
            #Mulai koneksi ke websocket inference server
            async with websockets.connect(shared.inference_addr, ping_timeout=10) as connection_ws:
                logger.info("Connected to inference server for health monitoring.")
                shared.inference_connected = True
                await send_to_app({"type": "inference_ok"})
                
                # Setelah terkoneksi selalu mengecek koneksi dengan ping tiap 5 detik
                try:
                    while shared.is_running:
                        await connection_ws.ping()
                        await asyncio.sleep(5)
                        
                except (websockets.exceptions.ConnectionClosed, 
                        websockets.exceptions.ConnectionClosedError,
                        asyncio.TimeoutError) as e:
                    logger.warning(f"Inference server connection lost: {e}")
                    shared.inference_connected = False
                    await send_to_app({"type": "inference_disconnect", "message": "Inference server disconnected"})
                    
        except Exception as e:
            if shared.inference_connected: 
                logger.warning(f"Failed to maintain inference connection: {e}")
                shared.inference_connected = False
                await send_to_app({"type": "inference_disconnect", "message": "Inference server disconnected"})
            
            await asyncio.sleep(5)
            
# Handler koneksi utama ke app & server 
async def connection_handler(ws):
    # Create UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.settimeout(3)  # 3 seconds timeout
    logger.info("App connected.")
    shared.connected_app = ws
    logger.info(f"New connection from {ws.remote_address} (ID: {id(ws)})")
    client = VoiceClient()

    try:
        # Send ready ke app untuk update status koneksi dengan controller
        await send_to_app({"type": "ready"})
        # Jalankan UDP discovery hanya akan lanjut jika berhasil ketemu server
        await udp_discovery(sock)
        
        # Mulai task monitoring koneksi inference server
        health_task = asyncio.create_task(maintain_inference_connection())
        
        # Create tasks for monitoring and recording
        tof_task = None
        recording_task = None
        
        # Main operation loop selama shared.is_running = True
        while shared.is_running:
            # Jika inference server sudah terhubung
            if shared.inference_connected:
                # Mulai background task monitor tof dan recording
                if tof_task is None or tof_task.done():
                    tof_task = asyncio.create_task(monitor_tof_sensor(client))
                if recording_task is None or recording_task.done():
                    recording_task = asyncio.create_task(continuous_recording(client))
                
                # Just wait and let the health monitor handle connection status
                await asyncio.sleep(1)  # Check every second instead of creating connections
                
                # Jika inference lost saat dicek di task cek inference
                if not shared.inference_connected:
                    logger.warning("Inference server disconnected!")
                    # Cancel semua task kalau disconnect ke server
                    if tof_task:
                        tof_task.cancel()
                        tof_task = None
                    if recording_task:
                        recording_task.cancel()
                        recording_task = None
            else:
                # Wait for connection to be restored by health monitor
                await asyncio.sleep(1)

    except websockets.exceptions.ConnectionClosed:
        logger.warning("App disconnected.")
    except Exception as e:
        logger.error(f"Error in app handler: {e}")
    finally:
        # Clean up task kalau app disconnect dari exception diatas
        shared.connected_app = None
        # Clean up tasks
        if 'health_task' in locals() and not health_task.done():
            health_task.cancel()
        if tof_task and not tof_task.done():
            tof_task.cancel()
        if recording_task and not recording_task.done():
            recording_task.cancel()

async def main():
    logger.info(f"App backend WebSocket server listening on ws://0.0.0.0:{APP_PORT}")
    server = await websockets.serve(connection_handler, "0.0.0.0", APP_PORT)

    try:
        await server.wait_closed()
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
        shared.is_running = False
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        shared.is_running = False