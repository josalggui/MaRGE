"""
:author:    J.M. Algarín
:email:     josalggui@i3m.upv.es
:affiliation: MRILab, i3M, CSIC, Valencia, Spain

"""
import numpy as np

from marge.widgets.widget_plot1d import Plot1DWidget


class Plot1DController(Plot1DWidget):
    """
    1D plot controller class.

    This class extends the `Plot1DWidget` class and serves as a controller for a 1D plot. It initializes the plot with
    the provided data and handles mouse movement events to display information about the data at the cursor position.

    Methods:
        __init__(self, x_data, y_data, legend, x_label, y_label, title): Initialize the Plot1DController instance.
        mouseMoved(self, evt): Handle the mouseMoved event to display information about the data at the cursor position.

    Attributes:
        y_data: The y data for the plot.
        x_data: The x data for the plot.
        x_label: The label for the x axis.
        y_label: The label for the y axis.
        title: The title of the plot.
        lines: A list of LineItems representing the plotted lines.
        plot_item: The PlotItem representing the plot.
        pen: A list of QPen objects representing the line colors.

    """

    def __init__(self,
                 x_data,  # numpy array
                 y_data,  # list of numpy array
                 legend,  # list of strings
                 x_label,  # string
                 y_label,  # string
                 title,  # string
                 ):
        """
        Initialize the Plot1DController instance.

        This method initializes the Plot1DController instance by calling the constructor of the parent class (`Plot1DWidget`).
        It sets the provided data and creates the plot lines. It also sets the plot properties such as title and axis labels.

        Args:
            x_data: The x data for the plot (numpy array).
            y_data: The y data for the plot (list of numpy arrays).
            legend: The legend for each line (list of strings).
            x_label: The label for the x axis (string).
            y_label: The label for the y axis (string).
            title: The title of the plot (string).

        Returns:
            None
        """
        super(Plot1DController, self).__init__()
        self.y_data = y_data
        self.x_data = x_data
        self.x_label = x_label
        self.y_label = y_label
        self.title = title

        # Set text
        self.label2.setText("<span style='font-size: 8pt'>%s=%0.2f, %s=%0.2f</span>" % (x_label, 0, y_label, 0))

        # Add lines to plot_item
        n_lines = len(y_data)
        self.lines = []
        x_min = 0
        x_max = 0
        y_min = 0
        y_max = 0
        for line in range(n_lines):
            if type(x_data) is list:
                x = x_data[line]
            else:
                x = x_data.copy()
            y = y_data[line]
            self.lines.append(self.plot_item.plot(x, y, pen=self.pen[line], name=legend[line]))
            if line == 0:
                x_min = np.min(x)
                x_max = np.max(x)
            if np.min(x) < x_min:
                x_min = np.min(x)
            if np.max(x) > x_max:
                x_max = np.max(x)
            if np.min(y) < y_min:
                y_min = np.min(y)
            if np.max(y) > y_max:
                y_max = np.max(y)
            self.plot_item.setXRange(x_min, x_max, padding=0)
            self.plot_item.setYRange(y_min, y_max, padding=0)
            if y_min == y_max:
                self.plot_item.setYRange(-1, 1, padding=0)

        # Set the plot properties
        self.plot_item.setTitle("%s" % title)
        self.plot_item.setLabel('bottom', x_label)
        self.plot_item.setLabel('left', y_label)

    def mouseMoved(self, evt):
        """
        Handle the mouseMoved event and update the information displayed at the cursor position on the plot.

        Args:
            evt (QGraphicsSceneMouseEvent): The mouse event object.

        Returns:
            None
        """
        pos = evt[0]
        if self.plot_item.sceneBoundingRect().contains(pos):
            if type(self.x_data) is not list:
                curves = self.plot_item.listDataItems()
                x, y = curves[0].getData()
                mouse_point = self.plot_item.vb.mapSceneToView(pos)
                index = np.argmin(np.abs(self.x_data - mouse_point.x()))
                self.label2.setText("<span style='font-size: 8pt'>%s=%0.2f, %s=%0.2f</span>" % (self.x_label,
                                                                                                x[index],
                                                                                                self.y_label,
                                                                                                y[index]))
                self.crosshair_v.setPos(x[index])
                self.crosshair_h.setPos(y[index])
            else:
                mouse_point = self.plot_item.vb.mapSceneToView(pos)
                self.label2.setText("x = %0.4f, y = %0.4f" % (mouse_point.x(), mouse_point.y()))
                self.crosshair_v.setPos(mouse_point.x())
                self.crosshair_h.setPos(mouse_point.y())
