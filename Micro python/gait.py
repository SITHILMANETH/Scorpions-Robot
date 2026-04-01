import pyrtos
from servos import site_now

x = 60
z_down = -50
z_up = -30

def walk_cycle():

    site_now[0] = [x,40,z_up]
    site_now[1] = [x,0,z_down]
    site_now[2] = [x,40,z_down]
    site_now[3] = [x,0,z_up]


def gait_task():

    while True:

        walk_cycle()

        yield [pyrtos.timeout(0.6)]

        site_now[0] = [x,40,z_down]
        site_now[1] = [x,0,z_up]
        site_now[2] = [x,40,z_up]
        site_now[3] = [x,0,z_down]

        yield [pyrtos.timeout(0.6)]
