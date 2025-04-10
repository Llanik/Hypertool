from PyQt5.QtWidgets import QApplication, QFileDialog,QMessageBox
import h5py
import numpy as np

def open_hyp(default_path="",open_window=True):

    if open_window:
        app=QApplication([])
        filepath, _ = QFileDialog.getOpenFileName(None,"Ouvrir un hypercube",default_path)
        print(filepath)
    else:
        filepath=default_path

    if filepath[-3:] == 'mat':
        try :
            with h5py.File(filepath, 'r') as file:
                cube=file['cube']
                wavelengths = file['#refs#/d'][:].flatten()
                hypercube = file['#refs#/c'][:]  # shape: (121, 900, 500)
                hypercube = hypercube.transpose(2, 1, 0)  # -> (500, 900, 121)
                return wavelengths, hypercube

        except:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Critical)
            msg_box.setText("Matlab file save mode not supported \n Please contact us")
            msg_box.exec_()

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
    import matplotlib.pyplot as plt
    folder=r'C:\Users\Usuario\Documents\DOC_Yannick\Hyperdoc_Test\Archivo chancilleria/'
    spec_range='VNIR'
    sample='pleito_52a.mat'
    filepath=folder+spec_range+'/mat/'+sample
    # filepath=r'C:\Users\Usuario\Downloads/A4_2.mat'
    try:
        wl,hyp=open_hyp(filepath)
        plt.imshow(hyp[:, :, [10, 30, 50]]/np.max(hyp[:, :, [10, 30, 50]]))
        plt.axis('off')
        plt.title(f'{sample} - {spec_range}')
    except: print('Hyp not loaded')

