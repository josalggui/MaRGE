"""
:author:    J.M. Algarín
:email:     josalggui@i3m.upv.es
:affiliation: MRILab, i3M, CSIC, Valencia, Spain

"""
import numpy as np

from widgets.widget_smith_chart import PlotSmithChartWidget


class PlotSmithChartController(PlotSmithChartWidget):
    """
    TODO
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
        TODO
        """
        super(PlotSmithChartController, self).__init__()
        self.y_data = y_data
        self.x_data = x_data
        self.x_label = x_label
        self.y_label = y_label
        self.title = title

        # Set text
        self.label2.setText("<span style='font-size: 8pt'>%s=%0.2f, %s=%0.2f</span>" % (x_label, 0, y_label, 0))

        # Create here the smith chart:
        theta = np.linspace(0, 2*np.pi, 100)
        x0 = np.cos(theta)
        y0 = np.sin(theta)
        x1 = 0.25 + 0.5 * np.cos(theta)
        y1 = 0.5 * np.sin(theta)
        x_smith = np.concatenate((x0, x1), axis=0)
        y_smith = np.concatenate((y0, y1), axis=0)
        if type(x_data) is list:
            x_data.append(x_smith)
        else:
            x_data = [x_data]
            x_data.append(x_smith)
        y_data = [y_data].append(y_data)

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
            if line == 1:
                self.lines.append(self.plot_item.plot(x, y, pen=self.pen[line], name=legend[line], symbol='o'))
            else:
                self.lines.append(self.plot_item.plot(x, y, pen=self.pen[line], name=legend[line]))
            if x[0] < x_min:
                x_min = x[0]
            if x[-1] > x_max:
                x_max = x[-1]
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
