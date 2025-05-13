from PyQt5 import QtWidgets, QtCore

class HypercubeManager(QtCore.QObject):
    cubesChanged = QtCore.pyqtSignal(list)
    def __init__(self):
        super().__init__()
        self._paths = []
    def addCube(self, path: str):
        if path and path not in self._paths:
            self._paths.append(path)
            self.cubesChanged.emit(self._paths)
    def removeCube(self, index: int):
        if 0 <= index < len(self._paths):
            self._paths.pop(index)
            self.cubesChanged.emit(self._paths)
    def clearCubes(self):
        self._paths.clear(); self._cubes.clear()
        self.cubesChanged.emit(self._paths)
    def getCube_path(self, index: int):
        return self._paths[index]
    @property
    def paths(self):
        return list(self._paths)