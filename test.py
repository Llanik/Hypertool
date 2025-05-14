import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QScrollArea,
    QLabel, QTextEdit, QCheckBox
)
from PyQt5.QtCore import Qt

class MetadataEditor(QWidget):
    def __init__(self, metadata: dict):
        super().__init__()
        self.setWindowTitle("Metadata : Label ↔ QTextEdit")
        self.resize(500, 400)

        # stocke les deux widgets par clé
        self._labels = {}
        self._edits  = {}

        layout = QVBoxLayout(self)

        # 1) Checkbox pour activer le mode édition
        self.chk_edit = QCheckBox("Edit Metadata")
        self.chk_edit.toggled.connect(self._toggle_edit)
        layout.addWidget(self.chk_edit)

        # 2) Zone défilable
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        # 3) Conteneur interne à la ScrollArea
        container = QWidget()
        scroll.setWidget(container)
        vbox = QVBoxLayout(container)

        # 4) Pour chaque métadonnée : QLabel + QTextEdit (caché au départ)
        for key, val in metadata.items():
            # clé en titre
            lbl_key = QLabel(f"<b>{key}</b>")
            vbox.addWidget(lbl_key)

            # QLabel pour la valeur (lecture seule)
            lbl = QLabel(str(val))
            lbl.setWordWrap(True)
            vbox.addWidget(lbl)

            # QTextEdit pour éditer (caché par défaut)
            edit = QTextEdit()
            edit.setPlainText(str(val))
            edit.setVisible(False)
            edit.setFixedHeight(80)
            edit.setLineWrapMode(QTextEdit.WidgetWidth)
            vbox.addWidget(edit)

            # mémorisation
            self._labels[key] = lbl
            self._edits[key]  = edit

        vbox.addStretch()

    def _toggle_edit(self, editable: bool):
        """
        Quand on coche/décoche la case :
         - si editable=True on affiche les QTextEdit (et on alimente leur texte)
         - sinon on affiche les QLabel
        """
        for key in self._labels:
            lbl  = self._labels[key]
            edit = self._edits[key]

            if editable:
                # préparer le QTextEdit avec le texte courant du QLabel
                edit.setPlainText(lbl.text())
            # on masque/affiche
            lbl.setVisible(not editable)
            edit.setVisible(editable)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Exemples de métadonnées, dont une multiligne
    sample_meta = {
        "Description": "Ligne A\nLigne B\nLigne C",
        "Device":      "Spectro-XYZ",
        "Notes":       "• Point 1\n• Point 2\n• Point 3\nLong texte possible..."
    }

    w = MetadataEditor(sample_meta)
    w.show()
    sys.exit(app.exec_())
