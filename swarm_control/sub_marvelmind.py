import cv2
# import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import time
import threading

from . import MarvelmindHedge


class SubMM():

    def __init__ (self, agent_num = 1, orientation=[0], timeout = 1000, freq = 5, tty = "COM6"):

        self.agent_num = agent_num
        self.orientation = [float(x) for x in orientation]
        self.tty = tty
        self.freq = freq
        self.timeout = timeout

        self.hedge = MarvelmindHedge(tty = self.tty, adr=None, debug=False)
        self.sub = 0 
        self.n = self.timeout*self.freq

        self.curr_pos = np.zeros((self.agent_num, 2))
        self.curr_vel = np.zeros((self.agent_num, 2))
        self.curr_ang_pos = np.zeros(self.agent_num)
        self.curr_ang_vel = np.zeros(self.agent_num)
        self.pos = np.zeros((self.n, self.agent_num, 2))
        self.vel = np.zeros((self.n, self.agent_num, 2))
        self.ang_pos = np.zeros((self.n, self.agent_num))
        self.ang_vel = np.zeros((self.n, self.agent_num))
        
        self.pos_i = [-1] * self.agent_num
        self.vel_i = [-1] * self.agent_num
        self.ang_pos_i = [0] * self.agent_num
        self.ang_vel_i = [0] * self.agent_num

        self.get_gyroscope_cal()
        


    def get_gyroscope_cal(self):
        self.gyro_cal = np.zeros((self.agent_num, 2,1))
        for i in range(1, self.agent_num+1):
            filename = 'calibration/beacon/' + str(i) + "_cal.yaml"
            cv_file = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ) 
            self.gyro_cal[i-1] = cv_file.getNode('gyro_cal').mat()


    def subscribe(self):
        self.sub = 1
        self.ang_pos[0] = self.orientation
        self.curr_ang_pos = np.array(self.orientation)
        self.thread = threading.Thread(target=self.get_info, daemon=True)
        self.hedge.start()
        self.thread.start()
        

    def unsubscribe(self):
        self.sub = 0 
        self.thread.join()
        self.hedge.stop()
    

    def get_info(self):

        imu_delta_t = [time.time()] * self.agent_num
        pos_delta_t = [time.time()] * self.agent_num
        while self.sub:

            self.hedge.dataEvent.wait(1)
            self.hedge.dataEvent.clear()

            if (self.hedge.positionUpdated):
                temp = self.hedge.position()
                hedge_id = temp[0]-1
                self.vel_i[hedge_id] += 1
                self.pos_i[hedge_id] += 1
                self.curr_vel[hedge_id][0] = (temp[1] - self.curr_pos[hedge_id][0])/(time.time() - pos_delta_t[hedge_id])
                self.curr_vel[hedge_id][1] = (temp[2] - self.curr_pos[hedge_id][1])/(time.time() - pos_delta_t[hedge_id])
                self.curr_pos[hedge_id][0] = temp[1]
                self.curr_pos[hedge_id][1] = temp[2]
                self.pos[self.pos_i[hedge_id]][hedge_id][0] = temp[1]
                self.pos[self.pos_i[hedge_id]][hedge_id][1] = temp[2]
                self.vel[self.vel_i[hedge_id]][hedge_id][0] = self.curr_vel[hedge_id][0]
                self.vel[self.vel_i[hedge_id]][hedge_id][1] = self.curr_vel[hedge_id][1]
                pos_delta_t[hedge_id] = time.time()
                time.sleep(1/(self.freq*self.agent_num*10))

            if (self.hedge.rawImuUpdated):
                temp = self.hedge.raw_imu()
                hedge_id = temp[0]-1
                self.ang_vel_i[hedge_id] += 1
                self.ang_pos_i[hedge_id] += 1
                self.curr_ang_vel[hedge_id] = (temp[6] - self.gyro_cal[hedge_id][0])*self.gyro_cal[hedge_id][1]
                self.curr_ang_pos[hedge_id] = self.ang_pos[self.ang_pos_i[hedge_id]-1][hedge_id] + self.curr_ang_vel[hedge_id] * (time.time()-imu_delta_t[hedge_id])
                if self.curr_ang_pos[hedge_id] > 180:
                    self.curr_ang_pos[hedge_id] -= 360
                elif self.curr_ang_pos[hedge_id] < -180:
                    self.curr_ang_pos[hedge_id] += 360
                self.ang_vel[self.ang_vel_i[hedge_id]][hedge_id] = self.curr_ang_vel[hedge_id]
                self.ang_pos[self.ang_pos_i[hedge_id]][hedge_id] = self.curr_ang_pos[hedge_id]
                imu_delta_t[hedge_id] = time.time()
                time.sleep(1/(self.freq*self.agent_num*10))

    

    def save_data(self):

        # get name of each columns
        pos_label = []
        vel_label = []
        ang_pos_label = []
        ang_vel_label = []
        for i in range(self.agent_num):
            id = str(i)
            pos_label.append('x_' + id)
            pos_label.append('y_' + id)
            vel_label.append('Vx_' + id)
            vel_label.append('Vy_' + id)
            ang_pos_label.append('theta_' + id)
            ang_vel_label.append('w_' + id)

        # change data form
        pos_data = pd.DataFrame(self.pos.reshape(self.n, self.agent_num*2), columns=pos_label)
        vel_data = pd.DataFrame(self.vel.reshape(self.n, self.agent_num*2), columns=vel_label)
        ang_pos_data = pd.DataFrame(self.ang_pos, columns=ang_pos_label)
        ang_vel_data = pd.DataFrame(self.ang_vel, columns=ang_vel_label)

        # determining the name of the file
        os.makedirs("data", exist_ok=True)
        curr = time.localtime(time.time())
        file_name = f'data\data_{curr[0]}-{curr[1]}-{curr[2]}-{curr[3]}{curr[4]}.xlsx'

        # creating an ExcelWriter object
        with pd.ExcelWriter(file_name) as writer:

            pos_data.to_excel(writer, sheet_name='Position')
            vel_data.to_excel(writer, sheet_name='Velocity')
            ang_pos_data.to_excel(writer, sheet_name='Angular Position')
            ang_vel_data.to_excel(writer, sheet_name='Angular Velocity')

            agent_label = ["x", "y", "Vx", "Vy", "theta", "w"]
            for i in range(self.agent_num):
                agent_array = np.transpose(np.vstack((self.pos[:,i,0], self.pos[:,i,1], self.vel[:,i,0], self.vel[:,i,0],
                                              self.ang_pos[:,i], self.ang_vel[:,i])))
                agent_data = pd.DataFrame(agent_array.reshape(self.n, 6), columns=agent_label)
                agent_data.to_excel(writer, sheet_name=f'Agent_{i}')

        print('DataFrames are written to Excel File successfully.')
        


    def gyro_calibration(self):

        self.hedge.start()
        gyro_cal = np.zeros(2)
        gyro_z = []
        hedge = -1

        t = time.time()
        while time.time() - t < 1:
            self.hedge.dataEvent.wait(1)
            self.hedge.dataEvent.clear()
            if (self.hedge.rawImuUpdated):
                hedge = self.hedge.raw_imu()[0]

        t = time.time()
        print("Caliberation Starting: keep the beacon at a flat and fixed location")
        while time.time() - t < 60:
            self.hedge.dataEvent.wait(1)
            self.hedge.dataEvent.clear()
            if (self.hedge.rawImuUpdated):
                gyro_z.append(self.hedge.raw_imu()[6]) 
        gyro_cal[0] = sum(gyro_z)/len(gyro_z)

        gyro_z_sum = np.zeros((4,1))
        if "s" == input("Rotate hedge 90 degree clockwise! (input 's' when ready)"):
            t = time.time()
            delta_t = time.time()
            while time.time() - t < 10:
                self.hedge.dataEvent.wait(1)
                self.hedge.dataEvent.clear()
                if (self.hedge.rawImuUpdated):
                    gyro_z_sum[0] += (self.hedge.raw_imu()[6] - gyro_cal[0])*(time.time()-delta_t)
                    delta_t = time.time()

        if "s" == input("Rotate hedge 180 degree clockwise! (input 's' when ready)"):
            delta_t = time.time()
            t = time.time()
            while time.time() - t < 10:
                self.hedge.dataEvent.wait(1)
                self.hedge.dataEvent.clear()
                if (self.hedge.rawImuUpdated):
                    gyro_z_sum[1] += (self.hedge.raw_imu()[6] - gyro_cal[0])*(time.time()-delta_t)
                    delta_t = time.time()

        if "s" == input("Rotate hedge 180 degree counterclockwise! (input 's' when ready)"):
            t = time.time()
            delta_t = time.time()
            while time.time() - t < 10:
                self.hedge.dataEvent.wait(1)
                self.hedge.dataEvent.clear()
                if (self.hedge.rawImuUpdated):
                    gyro_z_sum[2] += (self.hedge.raw_imu()[6] + gyro_cal[0])*(time.time()-delta_t)
                    delta_t = time.time()

        if "s" == input("Rotate hedge 90 degree counterclockwise! (input 's' when ready)"):
            delta_t = time.time()
            t = time.time()
            while time.time() - t < 10:
                self.hedge.dataEvent.wait(1)
                self.hedge.dataEvent.clear()
                if (self.hedge.rawImuUpdated):
                    gyro_z_sum[3] += (self.hedge.raw_imu()[6] + gyro_cal[0])*(time.time()-delta_t)
                    delta_t = time.time()

        gyro_cal[1] = (-90/gyro_z_sum[0] + -180/gyro_z_sum[1] + 180/gyro_z_sum[2] + 90/gyro_z_sum[3])/4
        print("parameter:", gyro_cal)

        # store calibration parameter in file
        os.makedirs("calibration/beacon", exist_ok=True)
        filename = 'calibration/beacon/' + str(hedge) + "_cal.yaml"
        file = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
        file.write(name='gyro_cal', val=gyro_cal)
        file.release()
        print("File stored in " + filename)

        self.hedge.stop()

