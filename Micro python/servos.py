from machine import I2C, Pin
import math
import pyrtos
from pca9685 import PCA9685

i2c = I2C(0, scl=Pin(22), sda=Pin(21))
pwm = PCA9685(i2c)
pwm.freq(60)

servo_map = [
[0,1,2],
[4,5,6],
[8,9,10],
[12,13,14]
]

length_a = 55
length_b = 77.5
length_c = 27.5

site_now = [[60,40,-50] for _ in range(4)]

pi = 3.1415926


def set_pwm(channel,val):
    pwm.duty(channel,val)


def cartesian_to_polar(x,y,z):

    w = math.sqrt(x*x + y*y)
    v = w - length_c

    alpha = math.atan2(z,v)
    beta = math.acos((length_a**2 + length_b**2 - v**2 - z**2)/(2*length_a*length_b))
    gamma = math.atan2(y,x)

    alpha = alpha/pi*180
    beta = beta/pi*180
    gamma = gamma/pi*180

    return alpha,beta,gamma


def polar_to_servo(leg,a,b,g):

    a = int((850/180)*a)
    b = int((850/180)*b)
    g = int((580/180)*g)

    a = max(125,min(580,a))
    b = max(125,min(580,b))
    g = max(125,min(580,g))

    set_pwm(servo_map[leg][0],a)
    set_pwm(servo_map[leg][1],b)
    set_pwm(servo_map[leg][2],g)


def update_leg(i):

    x,y,z = site_now[i]

    a,b,g = cartesian_to_polar(x,y,z)

    polar_to_servo(i,a,b,g)


def servo_task():

    while True:

        for i in range(4):
            update_leg(i)

        yield [pyrtos.timeout(0.02)]


def init_robot():

    print("Servos ready")
