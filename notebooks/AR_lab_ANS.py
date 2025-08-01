

import cv2 as cv
import numpy as np
from std_msgs.msg import Int32MultiArray
import csv
import os

    
def rodrigues_to_euler(rvec):
    # Convert Rodrigues vector to rotation matrix
    rotation_matrix, _ = cv.Rodrigues(rvec)
    # Check for gimbal lock
    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    locked = sy < 1e-6

    if not locked:
        roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        roll = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
        yaw = 0

    return np.array([roll, pitch, yaw])

def find_relative_pose(pic):
        ids = Int32MultiArray() #Each AR tag has a number written on it and a unique color combo, this will contain info for each 
        arucoDict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_100)
        mtx = np.array([[1.29650191e+03, 0.00000000e+00, 4.14079631e+02],#intrinsic camera matrix, calibrate with
                            [0.00000000e+00, 1.29687850e+03, 2.26798449e+02],#https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
                            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        distortion = np.array([[2.56660190e-01, -2.23374068e+00, -2.54856563e-03,  4.09905103e-03,9.68094572e+00]])
        
        result = grab_tag(pic, arucoDict, mtx, distortion)#find and process the image

        if len(result)>1:
            rvec, tvec = result
            tvec_m = tvec*0.01#convert to meters, the 26.6 market size was in cm
            trans, orien = find_pos(rvec, tvec_m)#get position relative to AR tag
            print(f"Orientation: {orien}, Position: {trans}")
        else:
            print('not found')
        
def grab_tag(tag, arucoDict, mtx, distortion):
        # Preprocess
        if tag is None:
            print("Error: Image is None.")
            return []
        tag = cv.cvtColor(tag, cv.COLOR_BGR2GRAY)
        
        corners, ids, rejects = cv.aruco.detectMarkers(tag, arucoDict) #find corners of the AR tag
        # https://docs.opencv.org/4.x/d9/d6a/group__aruco.html#gaba7f1e107f93451e2bc43b8ea96eef8c

        if len(corners) == 0:
            print('No corners found')
            return []
        
        rvec, tvec = my_estimatePoseSingleMarkers2(corners, mtx, distortion)#get stats from AR tag
        #rvec=rodrigues vector, tvec=translation vector of the AR tag in the camera frame

        cv.drawFrameAxes(image, mtx, distortion, rvec, tvec, length=10)#optional, helps visualization
        #red=x, y=green, z=blue
        print(f"rvecs: {rvec}, tvecs: {tvec}\n")
        return rvec, tvec
def find_pos(rvec1, tvec1):
        print(rvec1)
        rot_mat, jacobian = cv.Rodrigues(rvec1)
        wRd = np.transpose(rot_mat)#turns drone frame to world frame
        drone_from_ar = np.matmul(wRd, -tvec1)#finds translation vector from tag to drone in world frame
        #negative because tvec1 is from camera to tag, we want from tag to camera
        orien = rodrigues_to_euler(rvec1)
        return drone_from_ar, orien
def my_estimatePoseSingleMarkers2(corners, mtx, distortion, marker_size=26.6): # 26.6cm is side length of AR tag
        c= np.array(corners[0])[0]#corners of the AR tag in a numpy array in camera frame
        # https://github.com/Menginventor/aruco_example_cv_4.8.0/blob/main/pose_estimate.py


        marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],#corners of AR in world frame
                                [marker_size / 2, marker_size / 2, 0],
                                [marker_size / 2, -marker_size / 2, 0],
                                [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
        _, R, t = cv.solvePnP(marker_points, c, mtx, False, cv.SOLVEPNP_IPPE_SQUARE)
        return R, t

path = os.path.expanduser('~/BWSI_2025/Assignments/April_tags/AR_Images/AR_ims/image.png')
image = cv.imread(path, cv.IMREAD_COLOR)
if image is None:
    print("Failed to load image. Check the path.")

find_relative_pose(image)
cv.imshow("Pose Debug", image)
cv.waitKey(0)
cv.destroyAllWindows()