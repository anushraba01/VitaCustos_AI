import time
import board
import digitalio

IN1 = board.D17
IN2 = board.D18
IN3 = board.D22
IN4 = board.D23
ENA = board.D12
ENB = board.D13

in1 = digitalio.DigitalInOut(IN1)
in2 = digitalio.DigitalInOut(IN2)
in3 = digitalio.DigitalInOut(IN3)
in4 = digitalio.DigitalInOut(IN4)
ena = digitalio.DigitalInOut(ENA)
enb = digitalio.DigitalInOut(ENB)

in1.direction = digitalio.Direction.OUTPUT
in2.direction = digitalio.Direction.OUTPUT
in3.direction = digitalio.Direction.OUTPUT
in4.direction = digitalio.Direction.OUTPUT
ena.direction = digitalio.Direction.OUTPUT
enb.direction = digitalio.Direction.OUTPUT

ena.value = True
enb.value = True

def left_forward():
    in1.value = True
    in2.value = False

def left_backward():
    in1.value = False
    in2.value = True

def left_stop():
    in1.value = False
    in2.value = False

def right_forward():
    in3.value = True
    in4.value = False

def right_backward():
    in3.value = False
    in4.value = True

def right_stop():
    in3.value = False
    in4.value = False

def stop_all():
    left_stop()
    right_stop()

print("DC Motor Test with L298N")
print("Press Ctrl+C to stop\n")

try:
    print("Left motor: FORWARD")
    left_forward()
    time.sleep(3)
    
    print("Left motor: BACKWARD")
    left_backward()
    time.sleep(3)
    
    left_stop()
    time.sleep(1)
    
    print("Right motor: FORWARD")
    right_forward()
    time.sleep(3)
    
    print("Right motor: BACKWARD")
    right_backward()
    time.sleep(3)
    
    right_stop()
    time.sleep(1)
    
    print("Both motors: FORWARD")
    left_forward()
    right_forward()
    time.sleep(3)
    
    print("Both motors: BACKWARD")
    left_backward()
    right_backward()
    time.sleep(3)
    
    print("Test complete - stopping all motors")
    stop_all()

except KeyboardInterrupt:
    print("\nTest stopped by user")
    stop_all()