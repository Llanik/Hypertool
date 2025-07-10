import os
import sys
import importlib

def setup_matlab_env():
    base = r"C:\Program Files\MATLAB"
    if not os.path.isdir(base):
        return False

    versions = sorted(
        [v for v in os.listdir(base) if v.startswith("R")],
        reverse=True
    )

    for version in versions:
        base_path   = os.path.join(base, version)
        extern_bin  = os.path.join(base_path, "extern", "bin", "win64")
        core_bin    = os.path.join(base_path, "bin", "win64")
        python_eng  = os.path.join(base_path, "extern", "engines", "python")

        if all(os.path.isdir(p) for p in [extern_bin, core_bin, python_eng]):
            for p in [extern_bin, core_bin]:
                if p not in os.environ["PATH"]:
                    os.environ["PATH"] = p + os.pathsep + os.environ["PATH"]

            if python_eng not in sys.path:
                sys.path.insert(0, python_eng)

            return True
    return False

def load_matlab_engine():
    if not setup_matlab_env():
        raise RuntimeError("MATLAB environment not configured")

    try:
        matlab = importlib.import_module("matlab.engine")
    except ImportError as e:
        raise RuntimeError("Could not import matlab.engine") from e

    return matlab.start_matlab()
