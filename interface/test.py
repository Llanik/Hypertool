import joblib
import numpy as np
import glob
import os


for path in glob.glob(r"C:\Users\Usuario\Documents\GitHub\Hypertool\identification\data/*.joblib"):
    print('*******')
    print(path)
    model=joblib.load(path)
    print(model['pipeline'])


