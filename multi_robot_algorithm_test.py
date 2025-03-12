import numpy as np
import sys
import time

from robomaster import robot
from swarm_control import SubAruco, SubMM, Controller

Kp = 0.5
Kd = 0.3


def upper_bound(array, bound):
    
    sign = np.copy(array)
    sign[sign>=0] = 1
    sign[sign<0] = -1

    n = len(array)
    bounded_vel = [0]*n
    for i in range(n):
        bounded_vel[i] = min(abs(array[i]), bound)

    return bounded_vel * sign



def loc_to_glob(v, theta):
    '''
    v = np.array of size (2,1)
    theta = a float
    '''
    temp = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    coord_trans = np.linalg.inv(temp)
    result = np.matmul(coord_trans, v)
    return [result[0][0], result[1][0]]



def get_angle(vector):
    if vector.any():
        norm = np.sqrt(np.power(vector[0],2) + np.power(vector[1],2))
        vector = vector/norm
        angle = np.arccos(vector[0])
        if vector[1] >= 0:
            return angle
        else:
            return -angle
    else:
        return 0



def pd_control(id, s1_robot, s1_aruco, s1_mm):

    if s1_aruco.tvecs.any():
        print(f"robot{id} landmark detected:", s1_aruco.tvecs)
        print("theta:", s1_mm.curr_ang_pos[id])
        landmark_direction = s1_aruco.get_landmark_dir(s1_mm.curr_pos[id], s1_mm.ang_pos[id])
        print("landmark:", landmark_direction)


        target = np.sum(s1_aruco.tvecs, axis=0)/len(s1_aruco.tvecs)
        print("target,", target)
        if np.linalg.norm(target) > 0.3:
            vel = Kp * target + Kd * s1_mm.curr_vel[id]
            print(vel)
            bound_vel = upper_bound(vel, 0.2)
            s1_robot.chassis.drive_speed(bound_vel[0],bound_vel[1],0)
        else:
            print(f"Robot{id} done!")
            s1_control.mode[id] = 4
            s1_robot.chassis.drive_speed(0,0,0)
    
    else:
        s1_robot.chassis.drive_speed(0,0,0)



def run_algorithm(id, s1_robot, s1_aruco, s1_mm, s1_control):

    print(f"robot{id} landmark detected:", s1_aruco.tvecs)
    print("theta:", s1_mm.curr_ang_pos[id])
    landmark_direction = s1_aruco.get_landmark_dir(s1_mm.curr_pos[id], s1_mm.ang_pos[id])
    print("landmark:", landmark_direction)
    print("ang_vel:", s1_mm.curr_ang_vel[id])
    print("ang_pos:", s1_mm.curr_ang_pos[id])

    # check target
    if int(s1_mm.curr_ang_pos[id]/180*np.pi) == int(get_angle(landmark_direction)):
        print(f"Robot{id} done!")
        s1_control.mode[id] = 4
        s1_robot.chassis.drive_speed(0,0,0)
    else:
        torque = s1_control.pointing_behavior(s1_mm.curr_ang_vel[id], s1_mm.curr_ang_pos[id], landmark_direction)
        print("torque:", torque)
        ang_vel = int(upper_bound([torque], 3))
        # s1_robot.chassis.drive_speed(vel[0],vel[1],ang_vel)
        s1_robot.chassis.drive_speed(0,0,ang_vel)



def stop(s1_robots):
    for s1_robot in s1_robots:
        s1_robot.chassis.drive_speed(0,0,0)
    


if __name__ == "__main__":

    # setup the parameters for this process
    sn_list = ['159CG9V0050HED',  '159CKC50070ECX']
    # sn_list = ['159CG9V0050HED']
    agent_num = len(sn_list)
    time_out = 20
    orientation = [-90,-180]


    # connect to robot
    s1_robots = []
    s1_cameras = []
    for id in range(agent_num):
        s1_robot = robot.Robot()
        s1_robot.initialize(conn_type='sta', sn=sn_list[id])
        s1_robot.set_robot_mode(mode=robot.CHASSIS_LEAD)

        s1_gimbal = s1_robot.gimbal
        s1_gimbal.recenter()
        s1_camera = s1_robot.camera

        s1_robots.append(s1_robot)
        s1_cameras.append(s1_camera)
        print(f"Connect to Robot {id}")

    print("----------------- Connected to All Robots -----------------")


    # initialize marvelmind sensor subscription
    s1_mm = SubMM(agent_num=agent_num, orientation=orientation)
    s1_mm.subscribe()


    # initialize aruco marker subscription
    s1_arucos = []
    for i in range(agent_num):
        s1_aruco = SubAruco(s1_cameras[i], sn = sn_list[i], aruco_type="DICT_5X5_250", display=False)
        s1_aruco.subscribe()
        s1_arucos.append(s1_aruco)


    # initialize controller
    s1_control = Controller(agent_num=agent_num)


    # run algorithms in turn
    t = time.time()
    while time.time() - t < time_out and (s1_control.mode != 4).all():
        try:
            for i in range(agent_num):
                pd_control(i, s1_robots[i], s1_arucos[i], s1_mm)
                # test(i, s1_robots[i], s1_arucos[i], s1_mm, s1_control)
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Interrupt")
            stop(s1_robots)
            break

    # termination
    s1_mm.save_data()
    s1_mm.unsubscribe()
    for i in range(agent_num):
        s1_arucos[i].unsubscribe()
        s1_robots[i].close()

    print("----------------- Robots Closed -----------------")
    sys.exit()