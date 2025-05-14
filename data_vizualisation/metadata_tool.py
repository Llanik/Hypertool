from hypercubes.hypercube import*
from data_vizualisation.metadata_dock import*

class MetadataTool(QWidget, Ui_Metadata_tool):

    def __init__(self,cube_info:CubeInfoTemp, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.cube_info=cube_info

        # connect combobox
        self.comboBox_metadata.currentIndexChanged.connect(self.update_metadata_label)
        self.update_combo_meta(init=True)

        # stacked param init
        self.stacked_metadata.setCurrentIndex(0)
        self.lineEdit_metadata.setReadOnly(True)

        #connect to edit
        self.checkBox_edit.toggled.connect(self._toggle_edit_metadata)

    def update_combo_meta(self,init=False):

        last_key = self.comboBox_metadata.currentText()
        if last_key=='': last_key='cubeinfo'

        if init:
            self.comboBox_metadata.clear()

        if self.cube_info.metadata_temp is not None:
            for key in self.cube_info.metadata_temp.keys():
                if key not in ['wl','GT_cmap','spectra_mean','spectra_std']:
                    if key in ['GTLabels','pixels_averaged']:
                        try:
                            len(self.cube_info.metadata_temp[key])
                            self.comboBox_metadata.addItem(f"{key}")
                        except:
                            a=0

                    else:
                        self.comboBox_metadata.addItem(f"{key}")
                        if key==last_key:
                            self.comboBox_metadata.setCurrentText(key)

            self.update_metadata_label()

    def update_metadata_label(self):
        key = self.comboBox_metadata.currentText()
        if key=='':
            key='cubeinfo'
        raw = self.cube_info.metadata_temp[key]
        match key:
            case 'GTLabels':
                if len(raw.shape)==2:
                    st=f'GT indexes : <b>{(' , ').join(raw[0])}</b>  <br>  GT names : <b>{(' , ').join(raw[1])}</b>'
                elif len(raw.shape)==1:
                    st=f'GT indexes : <b>{(raw[0])}</b>  <br>  GT names : <b>{raw[1]}</b>'

            case 'aged':
                st=f'The sample has been aged ? <br> <b>{raw}</b>'

            case 'bands':
                st=f'The camera have <b>{raw[0]}</b> spectral bands.'

            case 'date':
                if len(raw)>1:info=raw
                else: info=raw[0]
                st=f'Date of the sample : <b>{info}</b>'

            case 'device':
                st=f'Capture made with the device : <br> <b>{raw}</b>'

            case 'illumination':
                st=f'Lamp used for the capture : <br> <b>{raw}</b>'

            case 'name':
                st=f'Name of the minicube : <br> <b>{raw}</b>'

            case 'number':
                st = f'Number of the minicube : <br> <b>{raw}</b>'

            case 'parent_cube':
                st = f'Parent cube of the minicube : <br> <b>{raw}</b>'

            case 'pixels_averaged':
                st = f'The number of pixels used for the <b>{len(raw)}</b> mean spectra of the GT materials are : <br> <b>{(' , ').join([str(r) for r in raw])}</b> '

            case 'reference_white':
                st = f'The reference white used for reflectance measurement is : <br> <b>{raw}</b>'

            case 'restored':
                st = f'The sample has been restored ?  <br> <b> {['NO','YES'][raw[0]]}</b>'

            case 'stage':
                st = f'The capture was made with a  <b>{raw}</b> stage'

            case 'reference_white':
                st = f'The reference white used for reflectance measurement is : <br> <b>{raw}</b>'

            case 'substrate':
                st = f'The substrate of the sample is : <br> <b>{raw}</b>'

            case 'texp':
                st = f'The exposure time set for the capture was <b>{raw[0]:.2f}</b> ms.'

            case 'height':
                st = f'The height of the minicube <b>{raw[0]}</b> pixels.'

            case 'width':
                st = f'The width of the minicube <b>{raw[0]}</b> pixels.'

            case 'position':
                st = f'The (x,y) coordinate of the upper right pixel of the minicube in the parent cube is : <br> <b>({raw[0]},{raw[1]})</b>'

            case 'range':
                val=['UV','VNIR : 400 - 1000 nm','SWIR : 900 - 1700 nm'][list(raw).index(1)]
                st = f'The range of the capture is : <br> <b>{val}</b>'

            case _:
                st=f'<b>{self.cube_info.metadata_temp[key]}</b>'

        self.label_metadata.setText(st)
        try : self.lineEdit_metadata.setText(raw)
        except : self.lineEdit_metadata.setText(repr(type(raw)))
        # TODO : affiche tous les metadata. faire des test comme pour affichage.

    def _toggle_edit_metadata(self, editable: bool):
        """
        Basculer entre mode lecture (QLabel) et mode édition (QLineEdit).
        """
        self.stacked_metadata.setCurrentIndex(1 if editable else 0)
        self.lineEdit_metadata.setReadOnly(not editable)
        self.update_metadata_label()

if __name__ == '__main__':
    sample   = '00001-VNIR-mock-up.h5'
    folder   = r'C:\Users\Usuario\Documents\DOC_Yannick\Hyperdoc_Test\Samples\minicubes'
    # sample = 'jabon_guillermo_final.mat'
    # sample = 'jabon_2-04-2025.mat'
    # folder = r'C:\Users\Usuario\Downloads'
    # sample = 'MPD41a_SWIR.mat'
    # folder = r'C:\Users\Usuario\Documents\DOC_Yannick\Hyperdoc_Test\Archivo chancilleria'
    filepath = os.path.join(folder, sample)

    cube = Hypercube(filepath=filepath, load_init=True)

    app = QApplication(sys.argv)

    window = MetadataTool(cube.cube_info)
    window.show()

    sys.exit(app.exec_())