import time
import board
import adafruit_dht
from luma.core.interface.serial import i2c
from luma.core.render import canvas
from luma.oled.device import ssd1306
from PIL import ImageFont, ImageDraw

DHT_PIN = board.D4
OLED_ADDRESS = 0x3C
SCREEN_WIDTH = 128
SCREEN_HEIGHT = 64
READ_INTERVAL = 2.0

try:
    dht_device = adafruit_dht.DHT22(DHT_PIN)
    print("DHT22 initialized on GPIO4")
except Exception as e:
    print(f" DHT22 init failed: {e}")
    
    exit(1)

try:
    serial = i2c(port=1, address=OLED_ADDRESS)
    oled = ssd1306(serial, width=SCREEN_WIDTH, height=SCREEN_HEIGHT)
    oled.clear()
    print("OLED initialized at address", hex(OLED_ADDRESS))
except Exception as e:
    print(f"OLED init failed: {e}")
    print("  Check I2C connection and enable I2C in raspi-config")
    exit(1)

print("\nDHT22 + OLED Monitor Running")
print("Press Ctrl+C to stop\n")

try:
    while True:
        try:
            temperature = dht_device.temperature
            humidity = dht_device.humidity
            
            if temperature is not None and humidity is not None:
                print(f"Temp: {temperature:.1f}°C | Humidity: {humidity:.1f}%")
                
                with canvas(oled) as draw:
                    draw.rectangle((0, 0, SCREEN_WIDTH-1, SCREEN_HEIGHT-1), outline=1)
                    
                    draw.text((32, 5), "DHT22 Monitor", fill=1)
                    draw.line((5, 15, 123, 15), fill=1)
                    
                    draw.text((10, 25), f"Temp: {temperature:.1f} C", fill=1)
                    
                    draw.text((10, 45), f"Humidity: {humidity:.1f} %", fill=1)
            
            else:
                print("Failed to read sensor - retrying")
            
            time.sleep(READ_INTERVAL)
            
        except RuntimeError as error:
            print(f"Sensor error (retrying): {error.args[0]}")
            time.sleep(READ_INTERVAL)
            continue
            
except KeyboardInterrupt:
    print("\n\nProgram stopped by user")
    dht_device.exit()
    oled.clear()
    print("Cleanup complete")