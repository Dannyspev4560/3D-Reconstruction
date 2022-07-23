import cv2 as cv
import glob
import numpy as np
#--------------------------calibrate--------------

class calibrate:
    def __init__(self):
        # Defining the dimensions of checkerboard
        CHECKERBOARD = (7, 7)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Creating vector to store vectors of 3D points for each checkerboard image
        objpoints = []
        # Creating vector to store vectors of 2D points for each checkerboard image
        imgpoints = []

        # Defining the world coordinates for 3D points
        objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        prev_img_shape = None
        # Extracting path of individual image stored in a given directory
        # images = glob.glob('./chess/*.jpg') - old
        # gray = 2
        images = glob.glob('chessboard/*.jpeg')
        for fname in images:
            img = cv.imread(fname)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            # gray = cv.resize(gray,(0,0), fx=0.4, fy=0.4)
            # cv.imshow('gray', gray)
            # cv.waitKey(0)
            # Find the chess board corners
            # If desired number of corners are found in the image then ret = true
            ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD,
                                                    cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)

            """
            If desired number of corner are detected,
            we refine the pixel coordinates and display 
            them on the images of checker board
            """
            if ret == True:
                objpoints.append(objp)
                # refining pixel coordinates for given 2d points.
                corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                imgpoints.append(corners2)

                # Draw and display the corners
                img = cv.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

            # cv.imshow('img', img)
            # cv.waitKey(0)

        cv.destroyAllWindows()

        # h, w = img.shape[:2]

        """
        Performing camera calibration by 
        passing the value of known 3D points (objpoints)
        and corresponding pixel coordinates of the 
        detected corners (imgpoints)
        """
        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    def get_calibration(self):
        return self.ret, self.mtx, self.dist, self.rvecs, self.tvecs