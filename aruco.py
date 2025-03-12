import cv2
import glob
import numpy as np
import os
import sys

from robomaster import robot


__all__ = ["ArucoMarker"]


class ArucoMarker():
    '''
    This class includes multiple different operations related aruco markers:
    generate_marker - generate .png files containing specified aruco marker in the folder 'marker' in workspace
    take_photo - need to connect to a robot through wifi; then, use the camera to take photos for calibration purpose
    camera_calibration - using the photo in folder 'calibration/sn' to get the parameters for the camera and calibrate the image
    detect_initialize - obtain parameters from calibration file
    pose_estimate - given frames of image, find the relative position of aruco markers
    '''

    def __init__(self, sn = "159CKC50070ECX", aruco_type="DICT_5X5_100", aruco_marker_side_length = 0.0245, max_marker=20):
        '''
            sn = the SN number of the robot detecting the markers
            aruco_type = the type of aruco marker used
            aruco_marker_side_length = the length of the printed aruco marker to be detected
            max_marker =  maximum number of markers a camera can detect
        '''
        self.aruco_type = aruco_type
        self.marker_size = int(aruco_type.split("_")[-1])
        self.max_marker = max_marker
        self.aruco_marker_side_length = aruco_marker_side_length
        self.sn = sn

        self.ARUCO_DICT =  {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
        "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
        "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
        "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
        "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
        "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
        "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
        "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
        "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
        "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
        "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
        "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL
        } 

        self.CameraMatrix = []
        self.DistortionMatrix = []
        self.marker_num = 0
        self.tvecs = np.zeros([1,2])
        self.rvecs = np.zeros([1,3])



    def generate_markers(self, num = 10):
        '''
        generate .png files containing specified aruco marker in the folder 'marker' in workspace
        num = number of markers generated
        '''

        arucoDict = cv2.aruco.getPredefinedDictionary(self.ARUCO_DICT[self.aruco_type])

        # create/load directory to store the marker
        offset = 0
        os.makedirs("markers", exist_ok=True)
        try:
            os.mkdir("markers/" + self.aruco_type)
        except FileExistsError:
            offset = len(os.listdir('markers/' + self.aruco_type))

        # generate marker
        for id in range(offset, num+offset):
            tag = np.zeros((self.marker_size, self.marker_size, 1), dtype='uint8')
            tag_name = 'markers/' + self.aruco_type + '/'+ self.aruco_type + '_' + str(id) + '.png'
            cv2.aruco.generateImageMarker(arucoDict, id, self.marker_size, tag, 1) # last argument denotes the padding around marker
            cv2.imwrite(tag_name, tag)
        
        print(f"{self.aruco_type} Markers Generated!")
    


    def take_photo(self, s1_robot, conn_type="ap"):
        '''
        connect to a robot; then, use the camera to take photos of the chessboard for calibration purpose

        conn_type = 'ap' - connect through wifi
                    'sta' - connect through router
        '''
        
        s1_camera = s1_robot.camera
        self.sn = str(s1_robot.get_sn())

        # setup image file location
        offset = 0
        os.makedirs("calibration", exist_ok=True)
        try:
            os.mkdir("calibration/" + self.sn)
        except FileExistsError:
            offset = len(os.listdir('calibration/' + self.sn))

        s1_camera.start_video_stream(display=True)
        i = offset

        while True:
            ret = input("enter 's+Enter' to take photo / enter 'q+Enter' when finished\n")
            if ret == "s":
                frame = s1_camera.read_video_frame()
                cv2.imwrite('calibration/' + self.sn + '/' + str(i) + '.png', frame)
                i += 1

            elif ret == "q":
                break
                
        s1_camera.stop_video_stream()
        print ("Photos stored in calibration folder")
    


    def camera_calibration(self, chessboardSize = (9,6), frameSize = (1280, 720)):
        '''
        using the photo in folder 'calibration/sn' to get the parameters for the camera and calibrate the image
        chessboardSize = size of the chessboard
        frameSize = frame size of the photos taken by the camera
        ''' 

        print("Calibration Starts...")

    ################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

        size_of_chessboard_squares_mm = 20
        objp = objp * size_of_chessboard_squares_mm


        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        images = glob.glob('calibration/' + self.sn + '/*.png')

        for image in images:

            img = cv2.imread(image)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, chessboardSize, None)

            # If found, add object points, image points (after refining them)
            if ret == True:

                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners)

                # Draw and display the corners
                cv2.drawChessboardCorners(img, chessboardSize, corners2, ret)
                cv2.imshow('img', img)
                cv2.waitKey(1000)

        cv2.destroyAllWindows()

    ############## CALIBRATION #######################################################
        ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

        filename = 'calibration/' + self.sn + "/" + self.sn + "_cal.yaml"
        file = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
        file.write(name='cameraMatrix', val=cameraMatrix)
        file.write(name='dist', val=dist)
        file.release()

    ############## UNDISTORTION #####################################################

        img = cv2.imread('calibration/' + self.sn + '/1.png')
        h,  w = img.shape[:2]
        newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))

        # Undistort
        dst = cv2.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        # cv2.imshow("undistored picture", dst)

        # Undistort with Remapping
        mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)
        dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        # cv2.imshow("undistored mapping picture", dst)

        # Reprojection Error
        mean_error = 0

        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error

        print( "Calibration Complete!\ntotal error: {}".format(mean_error/len(objpoints)) )

    

    def detect_initialize(self):
        '''
        initialize the detection of aruco markers: load the calibration parameters
        '''
        filename = 'calibration/' + self.sn + '/' + self.sn + "_cal.yaml"
        cv_file = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ) 
        self.CameraMatrix = cv_file.getNode('cameraMatrix').mat()
        self.DistortionMatrix = cv_file.getNode('dist').mat()
        cv_file.release()

        return self.CameraMatrix, self.DistortionMatrix



    def pose_estimation(self, frame):
        '''
        given frames of image, find the relative position of aruco markers
        frame = a frame of the video taken in numpy array format
        '''

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.aruco_dict = cv2.aruco.getPredefinedDictionary(self.ARUCO_DICT[self.aruco_type])
        parameters = cv2.aruco.DetectorParameters()

        corners, ids, _ = cv2.aruco.detectMarkers(gray, cv2.aruco_dict, parameters=parameters)

        if len(corners) > 0:

            self.marker_num = len(ids)
            self.tvecs = np.zeros([self.marker_num,2])
            self.rvecs = np.zeros([self.marker_num,3])

            for i in range(0, len(ids)):

                rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 
                        self.aruco_marker_side_length, self.CameraMatrix, self.DistortionMatrix)

                cv2.aruco.drawDetectedMarkers(frame, corners)
                cv2.drawFrameAxes(frame, self.CameraMatrix, self.DistortionMatrix, rvec, tvec, 0.05)
                
                self.tvecs[i] = [tvec[0][0][2], tvec[0][0][0]]
                self.rvecs[i] = rvec[0][0]
                # print(f"Marker {ids[i]}: Position = ({self.tvecs[i][0]:.2f}, {self.tvecs[i][1]:.2f}, {self.tvecs[i][2]:.2f}) meters")
        else:
            self.tvecs = np.zeros([1,2])
            self.rvecs = np.zeros([1,3])

        return frame

if __name__ == "__main__":
    # take photos of the chessboard and calibrate the camera 
    sn = "159CKC50070ECX"
    aruco = ArucoMarker(aruco_type="DICT_5X5_100", sn=sn)
    aruco.take_photo()
    aruco.camera_calibration()