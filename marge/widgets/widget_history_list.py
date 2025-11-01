"""
@author:    José Miguel Algarín
@email:     josalggui@i3m.upv.es
@affiliation:MRILab, i3M, CSIC, Valencia, Spain
"""
import sys

from PyQt5.QtWidgets import QListWidget, QWidget, QHBoxLayout, QCheckBox, QLabel, QLineEdit, QSizePolicy, QApplication, \
    QListWidgetItem


class HistoryListWidget(QListWidget):
    def __init__(self, parent, *args, **kwargs):
        super(HistoryListWidget, self).__init__(*args, **kwargs)
        self.main = parent
        self.setMaximumHeight(200)

    def addCustomItem(self, text):
        item = QListWidgetItem()
        widget = CustomItemWidget(text)
        item.setSizeHint(widget.sizeHint())
        self.addItem(item)
        self.setItemWidget(item, widget)

    def setCustomItemText(self, text, row=None):
        """
        Update the label text of a custom item.
        If 'row' is not provided, modifies the currently selected item.
        """
        # Determine which item to modify
        if row is None:
            item = self.currentItem()
            if not item:
                return  # no selection
        else:
            item = self.item(row)
            if not item:
                return  # invalid index

        # Retrieve the associated custom widget
        widget = self.itemWidget(item)
        # --- Update label text ---
        widget.label.setText(text)
        widget.label.adjustSize()  # update label's preferred width

        # --- Update list item height and layout ---
        item.setSizeHint(widget.sizeHint())  # resize the row
        self.updateGeometries()  # refresh internal layout
        self.viewport().update()  # force repaint

    def getCustomItemText(self, row=None):
        """
        Return the text of the QLabel in the specified row (or selected item).
        Returns None if the row or item is invalid.
        """
        if row is None:
            item = self.currentItem()
            if not item:
                return None
        else:
            item = self.item(row)
            if not item:
                return None

        widget = self.itemWidget(item)
        if widget and hasattr(widget, "label"):
            return widget.label.text()
        return None

class CustomItemWidget(QWidget):
    def __init__(self, label_text):
        super().__init__()
        layout = QHBoxLayout(self)

        self.checkbox = QCheckBox()
        self.label = QLabel(label_text)
        self.edit = QLineEdit()

        # --- Make label adjust to its text only ---
        self.label.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        self.label.setWordWrap(False)
        self.label.adjustSize()  # ensure minimum width fits the text

        # --- Make the line edit take all remaining space ---
        self.edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        layout.addWidget(self.checkbox, 0)
        layout.addWidget(self.label, 0)
        layout.addWidget(self.edit, 1)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HistoryListWidget(app)
    window.show()
    window.addCustomItem("Test")
    sys.exit(app.exec_())


