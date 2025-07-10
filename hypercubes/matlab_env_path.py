import os
import sys

def add_matlab_engine_path():
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
            if python_eng not in sys.path:
                sys.path.insert(0, python_eng)

            for p in [extern_bin, core_bin]:
                if p not in os.environ["PATH"]:
                    os.environ["PATH"] = p + os.pathsep + os.environ["PATH"]

            return True
    return False
