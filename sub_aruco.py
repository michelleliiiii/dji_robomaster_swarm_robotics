import cv2
import numpy as np
import time
import threading

from .aruco import ArucoMarker



__all__ = ["SubAruco", "SubPos", "SubIMU"]


class SubAruco(ArucoMarker):
    '''
    This is a super class of the ArucoMarker, 
    including operations that subscribe to the aruco info in the background using thread.
    subscribe - subscribe to the estimated position of aruco markers and store in self.tvecs
    unsubscribe - unsubscribe to the estimated position of aruco markers
    get_landmark - function run by the thread to get the aruco info
    get_landmark_dir - using other info to calculate the landmark direction at the instant
    '''

    def __init__ (self, s1_camera, freq = 5, display = True, sn = "159CKC50070ECX", aruco_type="DICT_5X5_100", 
                  aruco_marker_side_length = 0.0245, max_marker=20, test=0):
        '''
        s1_camera = the object from robomaster.robot.camera
        freq = the frequency that the info is received
        display = whether the image captured by camera is displayed on screen
        '''
        
        super().__init__ (sn, aruco_type, aruco_marker_side_length, max_marker)
        self.s1_camera = s1_camera
        self.freq = int(freq)
        self.display = display
        self.test = test

        self.sub = 0
        self.rel_pos = None
        self.landmark_dir = np.zeros((1, 2))


    def subscribe(self):
        self.sub = 1
        self.detect_initialize()
        self.thread = threading.Thread(target=self.get_landmark, daemon=True)
        self.thread.start()
           
    
    def unsubscribe(self):
        self.sub = 0
        self.thread.join()
        cv2.destroyAllWindows()
        self.s1_camera.stop_video_stream()
    

    def get_landmark(self):

        self.s1_camera.start_video_stream(display=False)

        while self.sub:
        
            frame = self.s1_camera.read_video_frame(strategy="newest", timeout=0.5)
            output = self.pose_estimation(frame)

            if self.display:
                cv2.imshow('Estimated Pose', output)
                cv2.waitKey(1) 

            if self.test:
                print("tvec:", self.tvecs)
            time.sleep(1/self.freq)
    

    def get_landmark_dir(self, glob_pos, anguler_pos, directions=np.zeros(1), neighbours=np.zeros(1)):
        '''
        loc_pos = a numpy array of shape (1,2) that denotes the relative position of the agent
        glob_pos = a numpy array of shape (1,2) that denotes the global position of the agent, [x, y, theta]
        anguler_pos = a float number theta
        directions = a numpy array of shape (m, 1, 2) that denotes the landmark direction (1,2) of m agents 
        neighbors = a numpy array of shape (m, 1) that contains boolean value where 1 indicates that 
                    the corresponding agent is in communication zone
        '''
        
        if self.tvecs.any(): # marker not empty
            coord_trans = np.array([[np.cos(anguler_pos)[0], -np.sin(anguler_pos)[0]], [np.sin(anguler_pos)[0], np.cos(anguler_pos)[0]]])
            self.rel_pos = (np.matmul(coord_trans, self.tvecs.transpose())).transpose()
            n = len(self.tvecs)
            dir = np.sum(self.rel_pos, axis=0)/ n
            norm = np.linalg.norm(dir)  
            self.landmark_dir = dir / norm  # normalized direction
        
        elif neighbours.any():
            self.rel_pos = None
            k = np.sum(neighbours)
            temp = np.multiply(neighbours, directions)[0] ### may be changed
            dir = np.sum(temp, axis=0) / k
            norm = np.linalg.norm(dir)  
            self.landmark_dir = dir / norm  # normalized direction
        
        else:
            self.rel_pos = None
            self.landmark_dir = np.zeros((1, 2))

        return self.landmark_dir 
    
        


