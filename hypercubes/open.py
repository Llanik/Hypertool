from PyQt5.QtWidgets import QApplication, QFileDialog,QMessageBox
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas


def open_hyp(default_dir=""):

    filepath, _ = QFileDialog.getOpenFileName(None,"Ouvrir un hypercube",default_dir)
    print(filepath)
    if filepath[-3:] == 'mat':

        with h5py.File(filepath, 'r') as file:
            cube=file['cube']
            wavelengths = file['#refs#/d'][:].flatten()
            hypercube = file['#refs#/c'][:]  # shape: (121, 900, 500)
            hypercube = hypercube.transpose(2, 1, 0)  # -> (500, 900, 121)
            return wavelengths,hypercube

    elif filepath[-3:] == '.h5':
        with h5py.File(filepath, 'r') as f:
            hypercube = np.array(f['DataCube']).T
            metadata = {attr: f.attrs[attr] for attr in f.attrs}
            wavelengths = metadata['wl']
            return wavelengths, hypercube

    else :
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setText("Data format not supported")
        msg_box.exec_()

if __name__ == "__main__":
    app=QApplication([])
    filepath=r'C:\Users\Usuario\Downloads/A4_2.mat'
    try:
        wl,hyp=open_hyp(filepath)
        plt.imshow(hyp[:, :, [10, 10, 50]]/np.max(hyp[:, :, [10, 10, 50]]))
    except: print('Hyp not loaded')

