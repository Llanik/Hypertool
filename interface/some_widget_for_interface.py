from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QVBoxLayout, QProgressBar,QLabel,QDialog
import os

class LoadingDialog(QDialog):
    def __init__(self, message="Loading...", filename=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Please wait")
        self.setModal(True)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        layout = QVBoxLayout(self)

        label = QLabel(message)
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        if filename:
            label_file = QLabel(f"<i>{os.path.basename(filename)}</i>")
            label_file.setAlignment(Qt.AlignCenter)
            layout.addWidget(label_file)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # mode indéterminé
        layout.addWidget(self.progress)

        self.setFixedWidth(350)

