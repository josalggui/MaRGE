"""
@author:    José Miguel Algarín
@email:     josalggui@i3m.upv.es
@affiliation:MRILab, i3M, CSIC, Valencia, Spain
"""
import numpy as np
import pyqtgraph as pg


class Plot3DWidget(pg.ImageView):
    def __init__(self,
                 main,
                 data=np.random.randn(10, 50, 50),
                 x_label='',
                 y_label='',
                 title=''):

        # Save inputs into the self
        self.main = main
        self.data = data
        self.x_label = x_label
        self.y_label = y_label
        self.title = title
        self.img_resolution = None

        # Define the PlotItem to display the image
        view = pg.PlotItem()
        view.setLabel(axis='left', text=y_label)
        view.setLabel(axis='bottom', text=x_label)
        view.setTitle(title=title)

        # Create text item
        self.text_item = pg.TextItem()
        self.text_item.setText('', color='red')

        # Execute the parent init
        super(Plot3DWidget, self).__init__(view=view)

        # hide FOV button if not image
        self.ui.menuBtn.hide()
        if title == "Sagittal" or title == "Coronal" or title == "Transversal":
            self.ui.menuBtn.show()

        # Change the Norm button to FOV button
        self.ui.menuBtn.setText("FOV")
        self.ui.menuBtn.setCheckable(True)

        # Create ROI to get FOV
        self.roiFOV = pg.ROI([0, 0], [np.size(data, 1), np.size(data, 2)])
        self.roiFOV.addScaleHandle([1, 1], [0.5, 0.5])
        self.roiFOV.addRotateHandle([0, 0], [0.5, 0.5])
        self.roiFOV.setZValue(20)
        self.roiFOV.setPen('y')
        self.view.addItem(self.roiFOV)
        self.roiFOV.hide()
        self.roiFOV.sigRegionChangeFinished.connect(self.roiFOVChanged)

        # Modify ROI to get SNR
        self.roi.setPen('g')

        # Add image
        self.setImage(data)