from robomaster import robot
import time
import numpy as np
import matplotlib.pyplot as plt
from swarm_control.marvelmind import MarvelmindHedge
import time
import sys


# predefined variables
TARGET = [1.2,-0.2]
INITIAL_ORIENT = [1,0]
MAX_TRANSLATION_SPEED = 0.5
MAX_ROTATION_SPEED = 30
TIMEOUT = 20
CONN_TYPE = "sta"
FREQ = 5
CONTROLLER_MODE = 1                   
'''
0 = rotate to target orientation and move in straight line
1 = keep the current orientation and drift to the desired location
'''

# controller parameters
GAIN_P0Y = 0.1
GAIN_P0X = 1
GAIN_D0X = 0.01
GAIN_R0D = 100
GAIN_R0P = 1

GAIN_P1 = 0.6
GAIN_D1 = 0.4


def setup():

    print("------------connected-----------------")

    # recenter gimbal
    s1_gimbal = s1_robot.gimbal
    s1_gimbal.recenter()

    # subscribe to the information
    global t
    t = time.time()
    s1_chassis.sub_imu(FREQ, imu_info)
    s1_chassis.sub_position(0, FREQ, position_info)
    
    # wait for info
    time.sleep(0.5)

    return np.array([round(TARGET[0],1), round(TARGET[1],1),0])


def plot():
    t = np.arange(0, n)
    tx = np.zeros(n)
    tx[:] = target[0]
    ty = np.zeros(n)
    ty[:] = target[1]
    max_speed = np.zeros(n)
    max_speed[:] = MAX_TRANSLATION_SPEED

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12)) = plt.subplots(4, 3, sharey=True)
    ax1.plot(t, loc_pos[:,0])
    ax2.plot(t, loc_pos[:,1])
    ax3.plot(t, loc_pos[:,2])

    ax4.plot(t, glob_pos[:,0])
    ax4.plot(t, tx)
    ax5.plot(t, glob_pos[:,1])
    ax5.plot(t, ty)

    ax7.plot(t, imu[:,0])
    ax8.plot(t, imu[:,1])
    ax9.plot(t, imu[:,2])
    ax10.plot(t, imu[:,3])
    ax11.plot(t, imu[:,4])
    ax12.plot(t, imu[:,5])

    ax1.set_title("local position x")
    ax2.set_title("local position y")
    ax3.set_title("local position z")
    ax4.set_title("global position x")
    ax5.set_title("global position y")
    ax7.set_title("acc x")
    ax8.set_title("acc y")
    ax9.set_title("acc z")
    ax10.set_title("gyro x")
    ax11.set_title("gyro y")
    ax12.set_title("gyro z")

    plt.show()
   

def terminate():
    
    s1_chassis.drive_speed(0,0,0)
    print ("------------terminating-----------------")
    hedge.stop()  # stop and close serial port
    s1_chassis.unsub_position()
    s1_chassis.unsub_imu()
    s1_robot.close()
    print("--------------terminated-----------------")


def position_info(position_info):
        
    global curr_loc_pos, loc_pos, index
    curr_loc_pos[0] = position_info
    loc_pos[index] = position_info

    hedge.dataEvent.wait(1)
    hedge.dataEvent.clear()
    if (hedge.positionUpdated):
        reading = hedge.position()
        curr_glob_pos[0] = reading[1:4]
        glob_pos[index] = reading[1:4]
    index += 1
    print("glob_pos:", curr_glob_pos)


def imu_info(imu_info):
    
    try:
        global curr_imu, imu, orient, t
        curr_imu[0] = imu_info
        imu[index] = imu_info
        orient = rotate_vector(orient, -imu_info[5]*(time.time()-t))
        t = time.time()
        print("orientation:", orient)

    except KeyboardInterrupt:
        terminate()


def upper_bound(array, bound):
     
    sign = np.copy(array)
    sign[sign>=0] = 1
    sign[sign<0] = -1

    n = len(array)
    bounded_vel = [0]*n
    for i in range(n):
        bounded_vel[i] = min(abs(array[i]), bound)

    return bounded_vel * sign


def distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1]))


def normalize(array):
    norm = np.sqrt(np.power(array[0],2) + np.power(array[1],2))
    return array/norm


def rotate_vector(v, theta):
    x = v[0]*np.cos(theta) + v[1]*np.sin(theta)
    y = -v[0]*np.sin(theta) + v[1]*np.cos(theta)
    return [x,y]


def get_angle(vector):
    angle = np.arccos(vector[0])
    if vector[1] >= 0:
        return angle
    else:
        return -angle


def reach_target(target_orient, orientation):
    if round(target_orient[0],1) != round(orientation[0], 1):
        return False
    elif round(target_orient[1],1) != round(orientation[1], 1):
        return False
    return True


def rotate_control():
    global orient, s1_chassis, interrupt
    start_time = time.time()
    target_orient = normalize((target - curr_glob_pos)[0])
    print("target", target_orient)
    print("current orientation:", curr_glob_pos)

    while not reach_target(target_orient, orient) and time.time()- start_time < TIMEOUT:
        
        try:
            time.sleep(0.01)
            rotate = GAIN_R0D * (get_angle(orient)-get_angle(target_orient)) + GAIN_R0P * (curr_imu[0][5])
            # print("angle", get_angle(orient)-get_angle(target_orient))
            # print("derivative of angle", curr_imu[0][5])
            rotate = float(upper_bound([rotate], MAX_ROTATION_SPEED))
            print(rotate)
            s1_chassis.drive_speed(0,0,rotate)
        
        except KeyboardInterrupt:
            terminate()
            interrupt = 1
            print("KeyboardInterrupt")
            sys.exit()
            break

    s1_chassis.drive_speed(0,0,0)

    if time.time()- start_time > TIMEOUT:
        print("rotation time out")


def straight_line(x, start_pos):
    desired_orientation = normalize((target - start_pos))
    slope = desired_orientation[1]/desired_orientation[0]
    # print(slope)
    y = slope * x[0] + (start_pos[1]-slope*start_pos[0])
    return y


def straight_line_control():

    print("start staight line control")
    global interrupt
    start_time = time.time()
    start_pos = curr_glob_pos[0]
    delta_t = 0.02

    while (target != np.round(curr_glob_pos[0], 1)).any() and time.time()- start_time < TIMEOUT:
        
        try:
            t = time.time()
            time.sleep(0.01)
            y_vel = GAIN_P0Y * (straight_line(curr_glob_pos[0], start_pos)-curr_glob_pos[0][0])
            # print(straight_line(curr_glob_pos[0], start_pos))
            x_vel = GAIN_P0X * (-target[0] + curr_glob_pos[0][0]) #+ GAIN_D0X * (-curr_glob_pos[0][0] + glob_pos[index-1][0])/delta_t
            # print("control velocity:", x_vel, y_vel)
            velocity = upper_bound([x_vel, y_vel], MAX_TRANSLATION_SPEED)
            # print(velocity)
            s1_chassis.drive_speed(velocity[0], velocity[1], 0)
            delta_t = time.time() - t

        except KeyboardInterrupt:
            terminate()
            interrupt = 1
            print("KeyboardInterrupt")
            sys.exit()
            break
    
    s1_chassis.drive_speed(0,0,0)
    if time.time()- start_time > TIMEOUT:
         print("staright line time out!")


def drift_control():

    print("start drift control")
    global interrupt
    start_time = time.time()
    delta_t = 0.02

    while (target != np.round(curr_glob_pos[0], 1)).any() and time.time()- start_time < TIMEOUT:
        
        try:
            t = time.time()
            time.sleep(0.05)
            glob_velocity = GAIN_P1 * (target - curr_glob_pos[0]) + GAIN_D1 * (-curr_glob_pos[0] + glob_pos[index-1])/delta_t
            loc_velocity = rotate_vector(glob_velocity, get_angle(orient))
            bounded_velocity = upper_bound(loc_velocity, MAX_TRANSLATION_SPEED)
            s1_chassis.drive_speed(bounded_velocity[0], -bounded_velocity[1], 0)
            delta_t = time.time() - t
            print("velocity:",  bounded_velocity)
        
        except KeyboardInterrupt:
            terminate()
            interrupt = 1
            plot()
            print("KeyboardInterrupt")
            sys.exit()
            break
        
    s1_chassis.drive_speed(0,0,0)
    if time.time()- start_time > TIMEOUT:
         print("drift control time out!")


def find_target():

    interrupt = 0

    # move in straight line toward the target
    if CONTROLLER_MODE == 0:

        # turn to correct orientation
        rotate_control()
        time.sleep(0.5)
        # move toward the target in a straight line
        straight_line_control()

    else:
        drift_control()

    return interrupt



if __name__ == '__main__':

    # connect to robot
    s1_robot = robot.Robot()
    s1_robot.initialize(conn_type=CONN_TYPE, proto_type="tcp", sn="159CG9V0050HED")
    s1_robot.set_robot_mode(mode=robot.CHASSIS_LEAD)

    # define variables
    orient = np.array(normalize(INITIAL_ORIENT))
    curr_loc_pos = np.zeros((1,3))
    curr_glob_pos = np.zeros((1,3))
    curr_imu = np.zeros((1,6))
    n = TIMEOUT * FREQ
    loc_pos = np.zeros((n,3))
    glob_pos = np.zeros((n,3))
    imu = np.zeros((n,6))
    index = 0

    # initailize setup
    hedge = MarvelmindHedge(tty = "COM6", adr=None, debug=False) # create MarvelmindHedge thread
    hedge.start()
    s1_chassis = s1_robot.chassis
    target = setup() 

    # approach the target
    if find_target() != 1:
        print("final position:", curr_glob_pos)
        terminate()
        plot()
        sys.exit()
