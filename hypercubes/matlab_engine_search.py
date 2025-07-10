import os
import sys
import numpy as np
import importlib.util

from hypercubes import matlab_env_path
from hypercubes.matlab_loader import load_matlab_engine


def setup_matlab_engine():
    """Try to setup the MATLAB Engine environment and import it safely."""
    if not matlab_env_path.add_matlab_engine_path():
        print("⚠️ MATLAB not found or environment not configured.")
        return False

    # Check if matlab.engine is available
    spec = importlib.util.find_spec("matlab.engine")
    if spec is None:
        print("⚠️ matlab.engine not found in current Python environment.")
        return False

    return True

def load_mat_file_with_engine(filepath):
    eng = load_matlab_engine()

    try:
        eng.eval(f"load('{filepath}')", nargout=0)

        try:
            data = eng.eval("cube.DataCube", nargout=1)
        except:
            raise ValueError("Impossible to find cube.DataCube")

        try:
            wl = eng.eval("cube.Wavelength", nargout=1)
        except:
            wl = None

        try:
            metadata = eng.eval("Metadata", nargout=1)
        except:
            metadata = None

    finally:
        eng.quit()

    return np.array(data), np.array(wl).squeeze() if wl else None, metadata

def get_matlab_engine():
    """Return the matlab.engine module if available, else raise informative error."""
    if not setup_matlab_engine():
        raise RuntimeError("MATLAB Engine is not available. Run `install_requirements.py` first.")

    import matlab.engine
    return matlab.engine