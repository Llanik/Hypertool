
import matlab.engine
import numpy as np

def load_cube_with_matlab(filepath):
    eng = matlab.engine.start_matlab()
    eng.eval(f"load('{filepath}')", nargout=0)

    data_cube = eng.eval("cube.DataCube", nargout=1)
    wavelengths = eng.eval("cube.Wavelength", nargout=1)

    try:
        metadata = eng.eval("Metadata", nargout=1)
    except:
        metadata = None

    eng.quit()

    data_np = np.array(data_cube)
    wl_np = np.array(wavelengths).squeeze()

    return data_np, wl_np, metadata


folder=r'C:\Users\Usuario\Documents\DOC_Yannick\HYPERDOC Database'
fname='cube_arabe.mat'
import os
path=os.path.join(folder,fname)

data, wl, meta = load_cube_with_matlab(path)

print(data.shape)         # (H, W, bands)
print(wl.shape)           # (bands,)
print(meta.keys())        # dict avec les infos utiles

from cv2 import imshow,waitKey
img=data[:,:,0]
imshow('image',img)
waitKey(0)