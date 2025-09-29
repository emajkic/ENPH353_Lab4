#!/usr/bin/env python3

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi

import cv2
import sys
import numpy as np

# My class to be used to avoid having to calculate these param for the template image every time a frame is processed 
class SIFTImage:
    def __init__(self,filepath,sift):
        self.filepath = filepath
        self.image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE) #read the image in grayscale
        # deal with self.image not existing later...

        self.keypoints, self.descriptors = sift.detectAndCompute(self.image, None)

        h,w = self.image.shape
        self.corners = np.float32([[0,0],[0,h],[w,h],[w,0]]).reshape(-1,1,2)

class My_App(QtWidgets.QMainWindow):
    def __init__(self):
        super(My_App, self).__init__()
        loadUi("./SIFT_app.ui", self)

        self._cam_id = 0
        self._cam_fps = 2
        self._is_cam_enabled = False
        self._is_template_loaded = False

        self.browse_button.clicked.connect(self.SLOT_browse_button)
        self.toggle_cam_button.clicked.connect(self.SLOT_toggle_camera)

        self._camera_device = cv2.VideoCapture(self._cam_id)
        self._camera_device.set(3, 320)
        self._camera_device.set(4, 240)

        # Timer used to trigger the camera
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.SLOT_query_camera)
        self._timer.setInterval(1000 / self._cam_fps)

        #Adding member vars -- MY CHANGE
        self._template_image = None
        self.sift = cv2.SIFT_create()

    def SLOT_browse_button(self):
        dlg = QtWidgets.QFileDialog()
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        if dlg.exec_():
            self.template_path = dlg.selectedFiles()[0]

        pixmap = QtGui.QPixmap(self.template_path)
        self.template_label.setPixmap(pixmap)

        print("Loaded template image file: " + self.template_path)

        #update template loaded flag, create our template image -- MY CHANGE
        self._is_template_loaded = True
        self._template_image = SIFTImage(self.template_path,self.sift)


    # Source: stackoverflow.com/questions/34232632/
    def convert_cv_to_pixmap(self, cv_img):
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        height, width, channel = cv_img.shape
        bytesPerLine = channel * width
        q_img = QtGui.QImage(cv_img.data, width, height,
                             bytesPerLine, QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(q_img)

    def SLOT_query_camera(self):

        if not self._is_template_loaded or self._template_image is None:
            return

        ret, frame = self._camera_device.read()
        
        # TODO run SIFT on the captured frame

        # Obtain keypts and desc for template 
        template_kp = self._template_image.keypoints
        template_desc = self._template_image.descriptors

        # Process the frame - including computing it's features 
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray_frame_kp, gray_frame_desc = self.sift.detectAndCompute(gray_frame, None)

        # Set up index and search param (should this be put into SIFTImage?)
        index_param = dict(algorithm = 0, trees = 5)
        search_param = dict()
        flann = cv2.FlannBasedMatcher(index_param, search_param)

        # Find matches and filter out false positives
        matches = flann.knnMatch(template_desc, gray_frame_desc, k=2)

        filtered_matches = []

        for m,n in matches: 
            if m.distance < 0.6*n.distance:
                filtered_matches.append(m)
        
        # Find homography 

        if len(filtered_matches) > 10:
            query_pts = np.float32([template_kp[m.queryIdx].pt for m in filtered_matches]).reshape(-1, 1, 2)
            train_pts = np.float32([gray_frame_kp[m.trainIdx].pt for m in filtered_matches]).reshape(-1, 1, 2)
            
            matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
            matches_mask = mask.ravel().tolist()

            #perform the perspective transform
            corners = self._template_image.corners

            dst = cv2.perspectiveTransform(corners, matrix)

            cv2.polylines(frame, [np.int32(dst)], True, (255,0,0), 3)

            pixmap = self.convert_cv_to_pixmap(frame)

        else: 
            
            matches_shown = cv2.drawMatches(self._template_image.image, 
            template_kp, 
            gray_frame, 
            gray_frame_kp, 
            filtered_matches, gray_frame)

            cv2.imshow("Matches", matches_shown)

            pixmap = self.convert_cv_to_pixmap(matches_shown)

        self.live_image_label.setPixmap(pixmap)

        #end of my additions

    def SLOT_toggle_camera(self):
        if self._is_cam_enabled:
            self._timer.stop()
            self._is_cam_enabled = False
            self.toggle_cam_button.setText("&Enable camera")
        else:
            self._timer.start()
            self._is_cam_enabled = True
            self.toggle_cam_button.setText("&Disable camera")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myApp = My_App()
    myApp.show()
    sys.exit(app.exec_())

