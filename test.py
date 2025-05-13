# main_example.py
# Exemple minimal d'une application PyQt5 avec un HypercubeManager et deux docks
# Gestion d'une liste de Hypercubes via un QToolButton muni d'un menu hiérarchique

import sys
from PyQt5 import QtWidgets, QtCore
from interface.HypercubeManager import HypercubeManager

class GenericToolWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        self.lbl = QtWidgets.QLabel("Aucun cube sélectionné", self)
        layout.addWidget(self.lbl)
        self.manager = None

    def displayCube(self, cube):
        self.lbl.setText(f"Cube chargé : {cube}")

# 3) MainApp avec QToolButton muni d'un menu structuré
class MainApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HyperdocApp Example")
        self.resize(600, 400)

        # Central widget vide
        self.setCentralWidget(QtWidgets.QWidget())

        # Manager unique
        self.hc_manager = HypercubeManager()

        # Création des docks
        self.dock1 = self._add_dock("Outil 1", GenericToolWidget, QtCore.Qt.LeftDockWidgetArea)
        self.dock2 = self._add_dock("Outil 2", GenericToolWidget, QtCore.Qt.RightDockWidgetArea)

        # Barre d'outils "Hypercubes"
        tb = self.addToolBar("Hypercubes")

        # Bouton "Cubes" sans indicateur de menu
        self.cubeBtn = QtWidgets.QToolButton(self)
        self.cubeBtn.setText("Cubes")
        self.cubeBtn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.cubeBtn.setStyleSheet("QToolButton::menu-indicator { image: none; }")
        tb.addWidget(self.cubeBtn)

        # Création du menu hiérarchique
        self.cubeMenu = QtWidgets.QMenu(self)
        self.cubeBtn.setMenu(self.cubeMenu)

        # Action d'ajout multiple
        act_add = QtWidgets.QAction("Add Cube…", self)
        act_add.triggered.connect(self._on_add_cube)
        tb.addAction(act_add)

        # Mise à jour du menu à chaque modification
        self.hc_manager.cubesChanged.connect(self._update_cube_menu)
        self._update_cube_menu(self.hc_manager.paths)

    def _add_dock(self, title, WClass, area):
        widget = WClass(parent=self)
        dock = QtWidgets.QDockWidget(title, self)
        dock.setWidget(widget)
        dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetClosable |
            QtWidgets.QDockWidget.DockWidgetMovable |
            QtWidgets.QDockWidget.DockWidgetFloatable
        )
        self.addDockWidget(area, dock)
        return dock

    def _on_add_cube(self):
        # Sélection multiple de fichiers
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select Hypercubes", "", "All files (*.*)"
        )
        for path in paths:
            self.hc_manager.addCube(path)

    def _update_cube_menu(self, paths):
        """Met à jour le menu de cubes avec sous-menus et actions fonctionnelles."""
        self.cubeMenu.clear()
        for idx, path in enumerate(paths):
            # Sous-menu pour chaque cube
            sub = QtWidgets.QMenu(path, self)
            # Envoyer au dock1
            act1 = QtWidgets.QAction("Send to Outil 1", self)
            act1.triggered.connect(lambda checked, i=idx: self._send_to_dock(i, self.dock1))
            sub.addAction(act1)
            # Envoyer au dock2
            act2 = QtWidgets.QAction("Send to Outil 2", self)
            act2.triggered.connect(lambda checked, i=idx: self._send_to_dock(i, self.dock2))
            sub.addAction(act2)
            # Séparateur
            sub.addSeparator()
            # Supprimer de la liste
            act_rm = QtWidgets.QAction("Remove from list", self)
            act_rm.triggered.connect(lambda checked, i=idx: self.hc_manager.removeCube(i))
            sub.addAction(act_rm)
            # Ajouter sous-menu au menu principal
            self.cubeMenu.addMenu(sub)

    def _send_to_dock(self, idx, dock):
        """Envoie le cube sélectionné au dock spécifié."""
        cube = self.hc_manager.getCube_path(idx)
        dock.widget().displayCube(cube)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = MainApp()
    win.show()
    sys.exit(app.exec_())
