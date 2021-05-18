# Exemplary plot:
# https://gist.github.com/nicoddemus/da4a727aef09de0dd0cfd2d1a6043104

from PyQt5 import QtChart
from warnings import warn
import numpy as np

"""
(!) This class is obsolete
"""

class ExamplePlot(QtChart.QChartView):
    def __init__(self, xData: list,
                 yData: list,
                 xLabel: str,
                 yLabel: str) -> None:

        super(ExamplePlot, self).__init__()

        if len(xData) != len(yData):
            warn("Length of x and y data does not match.")
            return

        yData_norm = yData / max(yData)

        series = QtChart.QLineSeries()

        for x, y in zip(xData, yData_norm):
            series.append(x, y)

        x_min = min(xData)
        x_max = max(xData)
        y_min = min(yData_norm)
        y_max = max(yData_norm)
        print("x_min = {}, x_max = {}, y_min = {}, y_max = {}".format(x_min, x_max, y_min, y_max))

        # chart = QtChart.QChart()

        self.chart().addSeries(series)
        self.chart().createDefaultAxes()
        self.chart().axisX(series).setTitleText(xLabel)
        self.chart().axisY(series).setTitleText(yLabel)
        self.chart().axisY(series).setRange(y_min, y_max)
        self.show()
