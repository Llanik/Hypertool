# cd C:\Users\Usuario\Documents\GitHub\Hypertool\registration
# python -m PyQt5.uic.pyuic -o registration_window.py registration_window.ui

import sys
import numpy as np
import cv2
from IPython.core.display_functions import update_display
from PyQt5.QtWidgets import (
    QApplication, QWidget,QMainWindow, QVBoxLayout, QPushButton,
    QLabel, QFileDialog, QHBoxLayout, QMessageBox, QComboBox,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
)
from PyQt5.QtGui import QPixmap, QImage, QTransform
from PyQt5.QtCore import Qt, QPointF, QRectF
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from registration.registration_window import*
from hypercubes.open import*

def np_to_qpixmap(img):
    if len(img.shape) == 2:
        try:
            qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_Grayscale8)
        except:
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            qimg = QImage(img.tobytes(), img.shape[1], img.shape[0],img.shape[1], QImage.Format_Grayscale8)

    else:
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0], rgb_image.strides[0], QImage.Format_RGB888)
    return QPixmap.fromImage(qimg).copy()

def overlay_color_blend(fixed, aligned):
    blended = cv2.merge([
        cv2.normalize(fixed, None, 0, 255, cv2.NORM_MINMAX),
        cv2.normalize(aligned, None, 0, 255, cv2.NORM_MINMAX),
        cv2.normalize(fixed, None, 0, 255, cv2.NORM_MINMAX)
    ])
    return blended

def overlay_checkerboard(fixed, aligned, tile_size=20):
    result = np.zeros_like(fixed)
    for y in range(0, fixed.shape[0], tile_size):
        for x in range(0, fixed.shape[1], tile_size):
            if ((x // tile_size) + (y // tile_size)) % 2 == 0:
                result[y:y+tile_size, x:x+tile_size] = fixed[y:y+tile_size, x:x+tile_size]
            else:
                result[y:y+tile_size, x:x+tile_size] = aligned[y:y+tile_size, x:x+tile_size]
    return result

class ZoomableGraphicsView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene())
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

    def setImage(self, pixmap):
        self.scene().clear()
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene().addItem(self.pixmap_item)
        self.setSceneRect(QRectF(pixmap.rect()))

    def wheelEvent(self, event):
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor
        zoom = zoom_in_factor if event.angleDelta().y() > 0 else zoom_out_factor
        self.scale(zoom, zoom)

class RegistrationApp(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("Image Registration")

        self.fixed_cube = None
        self.moving_cube = None
        self.aligned_cube= None
        self.fixed_img = None
        self.moving_img = None
        self.aligned_img = None
        self.kp1 = None
        self.kp2 = None
        self.show_features = False

        self.cube=[self.fixed_cube,self.moving_cube]
        self.img=[self.fixed_img,self.moving_img]
        self.radioButton_one=[self.radioButton_one_ref,self.radioButton_one_mov]
        self.radioButton_whole=[self.radioButton_whole_ref,self.radioButton_whole_mov]
        self.slider_channel=[self.horizontalSlider_ref_channel,self.horizontalSlider_mov_channel]
        self.spinBox_channel=[self.spinBox_ref_channel,self.spinBox_mov_channel]
        self.label_img=[self.label_fixed,self.label_moving]

        self.pushButton_open_ref_hypercube.clicked.connect(self.load_fixed)
        self.pushButton_open_mov_hypercube.clicked.connect(self.load_moving)
        self.pushButton_register.clicked.connect(self.register_images)

        self.overlay_selector.currentIndexChanged.connect(self.update_display)

        self.viewer_aligned = ZoomableGraphicsView()
        self.image_layout.addWidget(self.viewer_aligned, stretch=1)
        self.setLayout(self.main_layout)

        self.label_fixed.setAlignment(Qt.AlignCenter)
        self.label_moving.setAlignment(Qt.AlignCenter)

        self.horizontalSlider_ref_channel.setEnabled(False)
        self.horizontalSlider_mov_channel.setEnabled(False)
        self.spinBox_ref_channel.setEnabled(False)
        self.spinBox_mov_channel.setEnabled(False)

        self.horizontalSlider_ref_channel.valueChanged.connect(self.update_images)
        self.horizontalSlider_mov_channel.valueChanged.connect(self.update_images)


        self.radioButton_whole_ref.toggled.connect(self.update_sliders)
        self.radioButton_whole_mov.toggled.connect(self.update_sliders)

    def update_sliders(self):
        if self.radioButton_whole_ref.isChecked():
            self.horizontalSlider_ref_channel.setEnabled(False)
            self.spinBox_ref_channel.setEnabled(False)
        else:
            self.horizontalSlider_ref_channel.setEnabled(True)
            self.spinBox_ref_channel.setEnabled(True)

        if self.radioButton_whole_mov.isChecked():
            self.horizontalSlider_mov_channel.setEnabled(False)
            self.spinBox_mov_channel.setEnabled(False)

        else:
            self.horizontalSlider_mov_channel.setEnabled(True)
            self.spinBox_mov_channel.setEnabled(True)

        self.update_images()

    def update_images(self):

        for i_mov in [0,1]:
            cube=self.cube[i_mov]
            if cube is not None:
                mode = ['one', 'whole'][self.radioButton_whole[i_mov].isChecked()]
                chan = self.slider_channel[i_mov].value()
                img = self.cube_to_img(cube, mode, chan)
                img = (img * 256 / np.max(img)).astype('uint8')

                if i_mov:
                    self.moving_img = img
                else:
                    self.fixed_img = img
                self.img = [self.fixed_img, self.moving_img]

                self.label_img[i_mov].setPixmap(np_to_qpixmap(img).scaled(300, 300, Qt.KeepAspectRatio))

    def load_cube(self,i_mov):

        fname, _ = QFileDialog.getOpenFileName(self, ['Load Fixed Cube','Load Moving Cube'][i_mov])
        if fname:
            if fname[-3:] in['mat', '.h5']:
                _, cube = open_hyp(fname, open_window=False)
                if i_mov:
                    self.moving_cube = cube
                else:
                    self.fixed_cube = cube

                self.cube = [self.fixed_cube, self.moving_cube]
                self.slider_channel[i_mov].setMaximum(cube.shape[2]-1)
                self.spinBox_channel[i_mov].setMaximum(cube.shape[2]-1)

                mode = ['one', 'whole'][self.radioButton_whole[i_mov].isChecked()]
                chan = self.slider_channel[i_mov].value()
                img = self.cube_to_img(cube, mode, chan)
                img =(img * 256 / np.max(img)).astype('uint8')
            else:
                img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)

            if i_mov:
                self.moving_img=img
            else:
                self.fixed_img = img

            self.img = [self.fixed_img, self.moving_img]

            self.label_img[i_mov].setPixmap(np_to_qpixmap(img).scaled(300, 300, Qt.KeepAspectRatio))

    def load_fixed(self):
        self.load_cube(0)

    def load_moving(self):
        self.load_cube(1)

    def cube_to_img(self,cube,mode,chan):
        if mode=='whole':
            return np.mean(cube, axis=2).astype(np.float32)
        elif mode=='one':
            return cube[:,:,chan]

    def register_images(self):
        if self.fixed_img is None or self.moving_img is None:
            QMessageBox.warning(self, "Error", "Please load both images first.")
            return

        method = self.method_selector.currentText()

        if method == "ORB":
            self.register_features(cv2.ORB_create(5000))
        elif method == "AKAZE":
            self.register_features(cv2.AKAZE_create())
        elif method == "SIFT":
            self.register_features(cv2.SIFT_create())
        elif method == "ECC":
            self.register_images_ecc()
        else:
            QMessageBox.warning(self, "Error", "Unknown method.")

    def register_features(self, detector):
        kp1, des1 = detector.detectAndCompute(self.fixed_img, None)
        kp2, des2 = detector.detectAndCompute(self.moving_img, None)

        if des1 is None or des2 is None:
            QMessageBox.warning(self, "Error", "Feature detection failed.")
            return

        self.kp1, self.kp2 = kp1, kp2

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) if detector != cv2.SIFT_create() else cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        matches = matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)

        transform_type = self.transform_selector.currentText()
        if transform_type == "Affine":
            matrix, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
            self.aligned_img = cv2.warpAffine(self.moving_img, matrix, (self.fixed_img.shape[1], self.fixed_img.shape[0]))
            self.aligned_cube = np.zeros_like(self.fixed_cube, dtype=np.float32)
            for k in range(self.moving_cube.shape[2]):
                self.aligned_cube[:,:,k] = cv2.warpAffine(self.moving_cube[:,:,k], matrix,
                                                  (self.fixed_img.shape[1], self.fixed_img.shape[0]))

        elif transform_type == "Perspective":
            matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
            self.aligned_img = cv2.warpPerspective(self.moving_img, matrix, (self.fixed_img.shape[1], self.fixed_img.shape[0]))
            self.aligned_cube = np.zeros_like(self.fixed_cube, dtype=np.float32)
            for k in range(self.moving_cube.shape[2]):
                self.aligned_cube[:,:,k] = cv2.warpPerspective(self.moving_cube[:,:,k], matrix,
                                                  (self.fixed_img.shape[1], self.fixed_img.shape[0]))
        else:
            QMessageBox.warning(self, "Error", "Unsupported transformation.")
            return

        self.update_display()

    def register_images_ecc(self):
        try:
            fixed_f = self.fixed_img.astype(np.float32) / 255
            moving_f = self.moving_img.astype(np.float32) / 255

            warp_mode = cv2.MOTION_AFFINE
            warp_matrix = np.eye(2, 3, dtype=np.float32)

            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-6)
            cc, warp_matrix = cv2.findTransformECC(fixed_f, moving_f, warp_matrix, warp_mode, criteria)

            self.aligned_img = cv2.warpAffine(self.moving_img, warp_matrix, (self.fixed_img.shape[1], self.fixed_img.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            self.update_display()
        except Exception as e:
            QMessageBox.warning(self, "ECC Error", str(e))

    def update_display(self):
        if self.fixed_img is None or self.aligned_img is None:
            return

        display_mode = self.overlay_selector.currentText()
        if display_mode == "Color":
            img = overlay_color_blend(self.fixed_img, self.aligned_img)
        elif display_mode == "Checkboard":
            img = overlay_checkerboard(self.fixed_img, self.aligned_img)
        else:
            img = self.aligned_img

        # Display the final aligned image
        self.viewer_aligned.setImage(np_to_qpixmap(img))

if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = RegistrationApp()
    window.show()
    app.setStyle('Fusion')

    sys.exit(app.exec_())
