from machine import I2C, Pin
from pca9685 import PCA9685
import time

# I2C setup
i2c = I2C(0, scl=Pin(22), sda=Pin(21))

# PCA9685
pwm = PCA9685(i2c)
pwm.freq(60)

print("Starting servo test")

# move servo
def move_servo(channel):

    print("Testing servo channel:", channel)

    pwm.duty(channel,200)
    time.sleep(1)

    pwm.duty(channel,350)
    time.sleep(1)

    pwm.duty(channel,500)
    time.sleep(1)

    pwm.duty(channel,350)
    time.sleep(1)

# test first 12 servos (robot legs)
for servo in range(12):

    move_servo(servo)

print("Servo test finished")
