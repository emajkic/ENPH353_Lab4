#!/usr/bin/env python3

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi

import cv2
import sys
import numpy as np

class My_App(QtWidgets.QMainWindow):

	def __init__(self):
		super(My_App, self).__init__()
		loadUi("./SIFT_app.ui", self)

		# init SIFT, flann matcher, and parameters
		self.sift_obj = cv2.SIFT_create() # create SIFT Object
		self.index_params = dict(algorithm=0, trees=5)
		self.search_params = dict()
		self.flann = cv2.FlannBasedMatcher(self.index_params, self.search_params)

		self._cam_id = 0
		self._cam_fps = 10
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

	def SLOT_browse_button(self):
		dlg = QtWidgets.QFileDialog()
		dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)

		if dlg.exec_():
			self.template_path = dlg.selectedFiles()[0]
			self.reference_image = cv2.imread(self.template_path, cv2.IMREAD_GRAYSCALE) #store the uploaded img for later use
			self.reference_kp, self.reference_desc = self.sift_obj.detectAndCompute(self.reference_image, None) # extract keypoints from ref image
			# Sanity check:
			# img = cv2.drawKeypoints(self.reference_image, self.reference_kp, None)
			# cv2.imshow("", img) 

		pixmap = QtGui.QPixmap(self.template_path)
		self.template_label.setPixmap(pixmap)

		print("Loaded template image file: " + self.template_path)

	# Source: stackoverflow.com/questions/34232632/
	def convert_cv_to_pixmap(self, cv_img):
		cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
		height, width, channel = cv_img.shape
		bytesPerLine = channel * width
		q_img = QtGui.QImage(cv_img.data, width, height, 
					bytesPerLine, QtGui.QImage.Format_RGB888)
		return QtGui.QPixmap.fromImage(q_img)

	def SLOT_query_camera(self):
		ret, frame = self._camera_device.read()

		# ---------- SIFT algorithm on captured frame ----------

		# grayscale the video frame
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		#extract keypoints and descriptors of frame
		frame_kp, frame_desc = self.sift_obj.detectAndCompute(gray_frame, None)

		#find descriptor matches using FLANN
		matches = self.flann.knnMatch(self.reference_desc, frame_desc, k=2)
		good_points = []
		thresh = 0.6 #tune if needed

		for ref_element, frame_element in matches:
			if ref_element.distance < thresh*frame_element.distance: 
				good_points.append(ref_element)

		# Homography
		homography_thresh = 10

		if len(good_points) > homography_thresh:			
			query_points = np.float32([self.reference_kp[ref_element.queryIdx].pt for ref_element in good_points]).reshape(-1,1,2)
			train_points = np.float32([frame_kp[ref_element.trainIdx].pt for ref_element in good_points]).reshape(-1, 1, 2)

			matrix, mask = cv2.findHomography(query_points, train_points, cv2.RANSAC, 5.0)
			#matches_mask = mask.ravel.tolist()

			# perspective transform
			height, width = self.reference_image.shape
			corners = np.float32([[0,0], [0, height], [width, height], [width, 0]]).reshape(-1, 1, 2)
			distortion = cv2.perspectiveTransform(corners, matrix)

			cv2.polylines(frame, [np.int32(distortion)], True, (255,0,0), 3) 
			pixmap = self.convert_cv_to_pixmap(frame)
		else:
			matches_frame = cv2.drawMatches(self.reference_image, self.reference_kp, gray_frame, frame_kp, good_points, gray_frame)		
			pixmap = self.convert_cv_to_pixmap(matches_frame)

		self.live_image_label.setPixmap(pixmap)

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