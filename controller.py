import numpy as np
import time



class Controller():

    def __init__ (self, agent_num):

        # variables
        self.mode = np.zeros(agent_num) # 0 - search, 1 - explore, 3 - chase, 4 - dock/finish
        self.landmark_dir = np.zeros((agent_num,2))
        self.prev_stability = np.zeros(agent_num)

        # hyperparameters
        self.mass = 200
        self.inertia = 55
        self.agent_num = 8
        self.x_bound = [-1.5, 4]
        self.y_bound = [0, -5.5]
        self.bnc_k = 2
        self.R_col = 1.5
        self.R_comm = 15
        self.R_anti = 15
        self.R_flk = 1.75
        self.period = 30
        self.col_eta = 5
        self.flk_eta = 1
        self.anti_eta = 5 * self.flk_eta
        self.eplison = 0.1

        # parameters
        self.pnt_Kp = 1
        self.pnt_Kd = 5
        self.agg_Kp = 0.5



    @ staticmethod
    def get_neighbours(R, curr_pos, id, all_pos):
        ret = np.zeros((len(all_pos), 1))
        for i in range(len(all_pos)):
            if i != id:
                distance = np.linalg.norm(all_pos[i] - curr_pos[i])
                if distance < R:
                    ret[i] = 1

        return ret


    def mode_switch(self, i, rel_pos, rel_vel, landmark):
        if self.mode[i] == 1:
            if len(landmark) >= 3:
                self.mode[i] = 2
        elif self.mode[i] == 2:
            if len(landmark) <= 3:
                self.mode[i] = 1
                return False
            stability = self.stability(rel_pos, rel_vel, i)
            if stability - self.prev_stability[i] <= self.eplison:
                self.prev_stability[i] = stability
                self.mode[i] = 3
        elif self.mode[i] == 3:
            pass
        elif self.mode[i] == 4:
                return True

        return False


    def pointing_behavior (self, ang_vel, ang_pos, landmark_dir):
        ex = np.array([1,0])
        zeta = np.arccos(np.dot(landmark_dir, ex)) - ang_pos/180*np.pi
        torque = int(-self.pnt_Kp * zeta - self.pnt_Kd * ang_vel)
        return torque
    

    def search (self, id, curr_pos, all_pos, vel):

        # aggregation behavior
        if not self.landmark_dir.any():
            if np.linalg.norm(vel) != 0:
                vel_unit = vel/np.linalg.norm(vel)
            else:
                vel_unit = vel
            F_agg = -self.agg_Kp * (vel_unit - self.landmark_dir)
        else:
            F_agg = 0

        # collision behavior
        F_col= np.zeros([2,1])
        if len(all_pos) > 1:
            col_agent = self.get_neighbours(self.R_col, curr_pos, id, all_pos)
            for i in range(len(all_pos)):
                if col_agent[i]:
                    rel_pos = all_pos[i] - curr_pos
                    norm = np.linalg.norm(rel_pos)
                    F_col[0] += self.col_eta * (1/norm - 1/self.R_col) * rel_pos[0] / np.power(norm, 3)
                    F_col[1] += self.col_eta * (1/norm - 1/self.R_col) * rel_pos[1] / np.power(norm, 3)

        # bouncing behavior
        F_bnc_x = np.zeros([2,1])
        if curr_pos[0] > self.x_bound[1]:
            F_bnc_x[0] = -self.bnc_k
            F_bnc_x[1] = self.bnc_k/10 * np.random.rand()
        elif curr_pos[0] < self.x_bound[0]:
            F_bnc_x[0] = self.bnc_k
            F_bnc_x[1] = self.bnc_k/10 * np.random.rand()
        
        F_bnc_y = np.zeros([2,1])
        if curr_pos[1] > self.y_bound[1]:
            F_bnc_y[1] = -self.bnc_k
            F_bnc_y[0] = self.bnc_k/10 * np.random.rand()
        elif curr_pos[1] < self.y_bound[0]:
            F_bnc_y[1] = self.bnc_k
            F_bnc_y[0] = self.bnc_k/10 * np.random.rand()

        F_bnc = F_bnc_x + F_bnc_y
        
        return F_agg + F_col + F_bnc


    def explore (self, id, curr_pos, all_pos, markers):

        # flocking behavior
        F_flk = np.zeros([2,1])
        if markers is not None:
            for i in range(len(markers)):
                norm = np.linalg.norm(markers[i])
                F_flk[0] += -self.flk_eta * (norm - self.R_flk) * markers[i][0] / norm
                F_flk[1] += -self.flk_eta * (norm - self.R_flk) * markers[i][1] / norm
        

        # anti-flocking behavior
        F_anti= np.zeros([2,1])
        if len(all_pos) > 1:
            anti_agent = self.get_neighbours(self.R_anti, curr_pos, id, all_pos)
            for i in range(len(all_pos)):
                if anti_agent[i]:
                    rel_pos = all_pos[i] - curr_pos
                    norm = np.linalg.norm(rel_pos)
                    F_anti[0] += self.anti_eta * (1/norm - 1/self.R_anti) * rel_pos[0] / np.power(norm, 3)
                    F_anti[1] += self.anti_eta * (1/norm - 1/self.R_anti) * rel_pos[1] / np.power(norm, 3)

        return F_flk + F_anti


    def chase(self):
        pass
    

    def stability (self, rel_pos, rel_vel, i):
        pass

