import time
import board
import pwmio
from adafruit_motor import servo

pwm = pwmio.PWMOut(board.D18, duty_cycle=2 ** 15, frequency=50)

my_servo = servo.Servo(pwm)

print("Servo Motor Test")
print("Positions: 0° -> 90° -> 180° -> 0°")
print("Press Ctrl+C to stop\n")

try:
    while True:
        print("Moving to 0°")
        my_servo.angle = 0
        time.sleep(2)
        
        print("Moving to 90°")
        my_servo.angle = 90
        time.sleep(2)
        
        print("Moving to 180°")
        my_servo.angle = 180
        time.sleep(2)
        
        print("Returning to 0°")
        my_servo.angle = 0
        time.sleep(2)
        
except KeyboardInterrupt:
    print("\nTest stopped")
    my_servo.angle = None
    pwm.deinit()