import os
import numpy as np
import h5py
from spectral.io import envi
from PyQt5.QtWidgets import QApplication, QFileDialog,QMessageBox
from PyQt5.QtWidgets import QDialog
from hypercubes.save_window import Ui_Save_Window
import cv2
from scipy.io import savemat

# TODO : return dictionnaire ou dataclass from load (with cube, wl, metadata, filepath) + traiter ce dictionnaire dans registration
# TODO : look at ENVI open with lowercase etc...
# TODO : hypercube class to enforce in all tool

def open_hyp(default_path: str = "",
                   open_dialog: bool = True):
    """
    Open a hyperspectral cube from a .mat, .h5/.hdf5 or ENVI (.hdr) file.

    Returns:
        wavelengths: np.ndarray or None
        cube       : np.ndarray shape (lines, samples, bands) or None
        metadata   : dict of metadata fields or None
    """
    # 1) Ask user to select a file if requested
    if open_dialog:
        app = QApplication.instance() or QApplication([])
        filepath, _ = QFileDialog.getOpenFileName(
            None,
            "Open hyperspectral cube",
            default_path,
            "Hypercube files (*.mat *.h5 *.hdr)"
        )
        if not filepath:
            return None, None, None
    else:
        filepath = default_path

    ext = os.path.splitext(filepath)[1].lower()

    # 2) MATLAB .mat (HDF5-based)
    if ext == ".mat":
        try:
            with h5py.File(filepath, "r") as f:
                # Your internal layout: '#refs#/d' → wavelengths, '#refs#/c' → data
                wavelengths = f["#refs#/d"][:].flatten()
                raw = f["#refs#/c"][:]               # shape e.g. (bands, samples, lines)
                cube = np.transpose(raw, (2, 1, 0))  # → (lines, samples, bands)
                metadata = {}
                return wavelengths, cube, metadata
        except Exception:
            QMessageBox.critical(
                None, "Error",
                "Unsupported MATLAB file structure.\nPlease contact support."
            )
            return None, None, None

    # 3) HDF5 .h5 / .hdf5
    elif ext in (".h5", ".hdf5"):
        try:
            with h5py.File(filepath, "r") as f:
                raw = f["DataCube"][:]               # e.g. (bands, samples, lines)
                cube = np.transpose(raw, (2, 1, 0))
                metadata = {k: f.attrs[k] for k in f.attrs}
                wavelengths = metadata.get("wl")
                return wavelengths, cube, metadata
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Failed to read HDF5 file:\n{e}")
            return None, None, None

    # 4) ENVI (.hdr + raw .dat/.img)
    elif ext == ".hdr":
        try:
            # Open the ENVI header (automatically finds the raw)
            img = envi.open(filepath)
            # Load the full cube into memory
            cube = img.load().astype(np.float32)  # shape: (lines, samples, bands)
            metadata = img.metadata.copy()

            # Extract wavelengths (key may be 'wavelength' or 'wavelengths')
            wl = metadata.get("wavelength", metadata.get("wavelengths"))
            if isinstance(wl, str):
                # Convert a string "{400,410,...}" into a numeric array
                wl = wl.strip("{}").split(",")
                wavelengths = np.array(wl, dtype=np.float32)
            else:
                wavelengths = np.array(wl, dtype=np.float32) if wl is not None else None

            return wavelengths, cube, metadata

        except Exception as e:
            QMessageBox.critical(None, "Error", f"Failed to read ENVI file:\n{e}")
            return None, None, None

    # 5) Unsupported extension
    else:
        QMessageBox.critical(
            None, "Unsupported format",
            f"The extension '{ext}' is not supported."
        )
        return None, None, None

class SaveWindow(QDialog, Ui_Save_Window):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        # Accept on Save
        self.pushButton_save_cube_final.clicked.connect(self.accept)
        # Reject on Cancel
        self.pushButton_Cancel.clicked.connect(self.reject)

    def closeEvent(self, event):
        """
        Si l'utilisateur ferme la fenêtre via la croix,
        on rejette aussi le dialogue.
        """
        self.reject()
        super().closeEvent(event)

    def get_options(self):
        # ... ta méthode existante ...
        opts = {}
        opts['cube_format']    = self.comboBox_cube_format.currentText()
        opts['save_both']      = self.radioButton_both_cube_save.isChecked()
        opts['crop_cube']      = self.checkBox_minicube_save.isChecked()
        opts['export_images']  = self.checkBox_export_images.isChecked()
        if opts['export_images']:
            opts['image_format']   = self.comboBox_image_format.currentText()
            opts['image_mode_rgb'] = self.radioButton_RGB_save_image.isChecked()
        else:
            opts['image_format']   = None
            opts['image_mode_rgb'] = False
        opts['modify_metadata'] = self.checkBox_modif_metadata.isChecked()
        return opts

def save_hdf5_cube(path: str,
                   cube: np.ndarray,
                   metadata: dict = None,
                   dataset_name: str = "DataCube"):
    cube=cube = np.transpose(cube, (2, 1, 0))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        dset = f.create_dataset(dataset_name, data=cube)
        if metadata:
            # On ajoute les métadonnées à la racine et au dataset
            for key, value in metadata.items():
                try:
                    # si c'est un scalaire ou un array simple
                    dset.attrs[key] = value
                except TypeError:
                    # sinon on sérialise en string JSON
                    import json
                    dset.attrs[key] = json.dumps(value)
            # Optionnel : stocker aussi au niveau du groupe racine
            for key, value in metadata.items():
                f.attrs[key] = dset.attrs[key]


def save_envi_cube(path_hdr: str,
                   cube: np.ndarray,
                   metadata: dict = None,
                   interleave: str = "bil",
                   dtype_code: int = 4):

    os.makedirs(os.path.dirname(path_hdr), exist_ok=True)
    hdr_meta = {
        "lines":      cube.shape[0],
        "samples":    cube.shape[1],
        "bands":      cube.shape[2],
        "data type":  dtype_code,
        "interleave": interleave
    }
    if metadata:
        # on fusionne metadata dans hdr_meta (clé->valeur)
        hdr_meta.update(metadata)
    envi.save_image(path_hdr, cube.astype(np.float32), metadata=hdr_meta)

def save_matlab_cube(path: str,
                     cube: np.ndarray,
                     metadata: dict = None,
                     varname: str = "cube"):

    os.makedirs(os.path.dirname(path), exist_ok=True)
    tosave = {varname: cube}
    if metadata:
        for key, value in metadata.items():
            tosave[key] = value
    savemat(path, tosave)

def save_images(dirpath: str,
                fixed_img: np.ndarray,
                aligned_img: np.ndarray,
                image_format: str = "png",
                rgb: bool = False):

    os.makedirs(dirpath, exist_ok=True)
    ext = image_format.lower()
    def write(name, img):
        out = os.path.join(dirpath, f"{name}.{ext}")
        if img.ndim == 2 and rgb:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.ndim == 3 and rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(out, img)
    write("fixed", fixed_img)
    write("aligned", aligned_img)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("TkAgg")

    sample='MPD41a_SWIR.mat'
    folder_cube=r'C:\Users\Usuario\Documents\DOC_Yannick\Hyperdoc_Test\Archivo chancilleria/'
    filepath=folder_cube+sample
    try:
        wl,hyp,meta=open_hyp(filepath)
        plt.figure()
        plt.imshow(hyp[:, :, [50, 30, 10]]/np.max(hyp[:, :, [50, 30, 10]]))
        plt.axis('off')
    except: print('Hyp not loaded')

    # app = QApplication.instance() or QApplication([])
    # file_path_save,_=QFileDialog.getSaveFileName (None,
    #         "Save hyperspectral cube")
    # save_hdf5_cube(file_path_save,cube=hyp,metadata=meta)

