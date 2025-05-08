from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QTimer,QSize, Qt
from PyQt5.QtGui import QFont,QIcon, QPalette, QColor
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QStyleFactory, QToolBar,QAction,QComboBox,QLabel

import sys
import os
import traceback

# widgets import
from hypercubes.hypercube      import HDF5BrowserWidget
from data_vizualisation.data_vizualisation_tool import Data_Viz_Window
from registration.register_tool          import RegistrationApp

# TODO : initier dans MainWindow les hypercubes et connecter les champs de chaque widget (yeah...big deal)

from PyQt5.QtWidgets import QToolBar, QDockWidget
from PyQt5.QtCore    import QSize, Qt
from PyQt5.QtGui     import QIcon

def apply_fusion_border_highlight(app,
                                  border_color: str = "#888888",
                                  title_bg:      str = "#E0E0E0",
                                  separator_hover: str = "#AAAAAA",
                                  window_bg:     str = "#F5F5F5",   # ← ton nouveau fond
                                  base_bg:       str = "#EFEFEF"):  # ← pour QTextEdit, etc.
    # 1) Fusion
    app.setStyle(QStyleFactory.create("Fusion"))

    # 1b) palette customisée
    pal = app.palette()
    pal.setColor(QPalette.Window,        QColor(window_bg))
    pal.setColor(QPalette.Base,          QColor(base_bg))
    app.setPalette(pal)

    # 2) ton QSS existant pour les bordures
    app.setStyleSheet(f"""
    QMainWindow, QWidget#centralwidget {{
        background-color: {window_bg};
    }}
    QDockWidget {{
        border: 1px solid {border_color};
    }}
    QDockWidget::title {{
        background: {title_bg};
        padding: 3px;
        border-bottom: 1px solid {border_color};
        color: black;
        text-align: left;
    }}
    QMainWindow::separator {{
        background-color: {border_color};
        width: 2px; height: 2px; margin: 1px;
    }}
    QMainWindow::separator:hover {{
        background-color: {separator_hover};
    }}
    QSplitter::handle {{
        background-color: {border_color};
    }}
    QSplitter::handle:hover {{
        background-color: {separator_hover};
    }}
    /* assure que les widgets enfants héritent bien de la couleur de fond */
    QDockWidget > QWidget {{
        background-color: {base_bg};
    }}
    """)


class MainApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HyperdocApp")
        self.resize(1200, 800)
        self.setCentralWidget(QtWidgets.QWidget())

        BASE_DIR = os.path.dirname(os.path.dirname(__file__))
        ICONS_DIR = os.path.join(BASE_DIR, "Hypertool/interface/icons")
        icon_main= "Hyperdoc_logo_transparente_CIMLab.png"
        self.setWindowIcon(QIcon(os.path.join(ICONS_DIR,icon_main)))

        # add docks
        self.file_browser_dock=self._add_dock("File Browser",   HDF5BrowserWidget,   QtCore.Qt.LeftDockWidgetArea)
        self.data_viz_dock =self._add_dock("Data Visualization", Data_Viz_Window,  QtCore.Qt.RightDockWidgetArea)
        self.reg_dock=self._add_dock("Registration",   RegistrationApp,     QtCore.Qt.BottomDockWidgetArea)

        # Tool menu
        view = self.menuBar().addMenu("Tools")
        for dock in self.findChildren(QtWidgets.QDockWidget):
            view.addAction(dock.toggleViewAction())

        # ─── Toolbar “Quick Tools” ───────────────────────────────────────────
        toolbar = self.addToolBar("Quick Tools")
        toolbar.setIconSize(QSize(48, 48))  # Taille des icônes
        toolbar.setToolButtonStyle(Qt.ToolButtonIconOnly)  #ToolButtonIconOnly ou TextUnderIcon)

        # Action File Browser
        act_file = self.file_browser_dock.toggleViewAction()
        icon_file_browser = "file_browser_icon.png"
        act_file.setIcon(QIcon(os.path.join(ICONS_DIR, icon_file_browser)))  # charge ton icône
        act_file.setToolTip("File Browser")
        toolbar.addAction(act_file)

        # Action Data Viz
        act_data = self.data_viz_dock.toggleViewAction()
        icon_data_viz = "icon_data_viz.svg"
        act_data.setIcon(QIcon(os.path.join(ICONS_DIR, icon_data_viz)))
        act_data.setToolTip("Data Visualization")
        toolbar.addAction(act_data)

        # Action Registration
        act_reg = self.reg_dock.toggleViewAction()
        icon_registration = "registration_icon.png"
        act_reg.setIcon(QIcon(os.path.join(ICONS_DIR, icon_registration)))
        act_reg.setToolTip("Registration")
        toolbar.addAction(act_reg)

        toolbar.addSeparator()

        # Action Open File and previous next
        act_open_file = QAction("Open file", self)
        act_reg.setToolTip("Open a file with hypercube")
        toolbar.addAction(act_open_file)

        act_open_previous = QAction("<", self)
        act_open_previous.setToolTip("Open previous cube in current folder")
        toolbar.addAction(act_open_previous)

        act_open_next = QAction(">", self)
        act_open_next.setToolTip("Open next cube in current folder")
        toolbar.addAction(act_open_next)

        toolbar.addSeparator()

        # Action Active Hyp : combobox
        label_active_hyp=QLabel("Active cube :")
        toolbar.addWidget(label_active_hyp)

        combo = QComboBox()
        combo.addItems(["Cube name 1", "Cube name 2"])
        combo.setCurrentIndex(1)
        act_reg.setToolTip("Active Cube")

        toolbar.addWidget(combo)

        # tools to hide at opening
        self.file_browser_dock.hide()
        self.reg_dock.hide()

    def _add_dock(self, title, WidgetClass, area):
        widget = WidgetClass(parent=self)
        dock   =  QtWidgets.QDockWidget(title, self)
        dock.setWidget(widget)
        self.addDockWidget(area, dock)
        return dock

def excepthook(exc_type, exc_value, exc_traceback):
    """Capture les exceptions et les affiche dans une boîte de dialogue."""
    error_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    print(f"Erreur : {error_msg}")  # Optionnel : enregistrer dans un fichier log
    msg_box = QtWidgets.QMessageBox()
    msg_box.setIcon(QtWidgets.QMessageBox.Critical)
    msg_box.setText("Une erreur est survenue :")
    msg_box.setInformativeText(error_msg)
    msg_box.exec_()

def update_font(_app,width=None,_font="Segoe UI",):
    global main

    if not width:
        screen = _app.primaryScreen()
        screen_size = screen.size()
        screen_width = screen_size.width()

    else:
        screen_width=width

    if screen_width < 1280:
        font_size = 7
    elif screen_width < 1920:
        font_size = 8
    else:
        font_size = 9

    _app.setFont(QFont(_font, font_size))
    plt.rcParams.update({"font.size": font_size + 3, "font.family": _font})

def check_resolution_change():
    """ Vérifie si la résolution a changé et met à jour la police si besoin """
    global last_width  # On garde la dernière largeur connue
    screen = app.screenAt(main.geometry().center())
    current_width = screen.size().width()

    if current_width != last_width:
        update_font(app,current_width)
        last_width = current_width

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    update_font(app)
    # apply_fusion_dark_theme(app)
    apply_fusion_border_highlight(app)
    # app.setStyle('Fusion')

    main = MainApp()
    main.show()

    # Timer for screen resolution check
    last_width = app.primaryScreen().size().width()
    timer = QTimer()
    timer.timeout.connect(check_resolution_change)
    timer.start(500)  # Vérifie toutes les 500 ms
    sys.exit(app.exec_())