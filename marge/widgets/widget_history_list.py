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
        self.setCurrentItem(item)
        self.scrollToItem(item)

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
        widget.label.setText(text)

    def allUnchecked(self):
        """Return True if all checkboxes are unchecked."""
        for i in range(self.count()):
            item = self.item(i)
            widget = self.itemWidget(item)
            if widget.checkbox.isChecked():
                return False
        return True

    def getCheckedItems(self):
        """Return a list with the label texts of all checked items."""
        checked_labels = []
        for i in range(self.count()):
            item = self.item(i)
            widget = self.itemWidget(item)
            if widget.checkbox.isChecked():
                checked_labels.append(widget.label.text().split('|')[1].split(' ')[1])
        return checked_labels

    def getCustomItemText(self, item=None):
        """
        Return the text of the QLabel in the specified row (or selected item).
        Returns None if the row or item is invalid.
        """
        if item is None:
            item = self.currentItem()
            if not item:
                return None

        widget = self.itemWidget(item)
        return widget.label.text()

class CustomItemWidget(QWidget):
    def __init__(self, label_text):
        super().__init__()
        layout = QHBoxLayout(self)

        self.checkbox = QCheckBox()
        self.label = QLabel(label_text)
        self.edit = QLineEdit()

        layout.addWidget(self.checkbox, 0)
        layout.addWidget(self.label, 3)
        layout.addWidget(self.edit, 1)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HistoryListWidget(app)
    window.show()
    window.addCustomItem("Test")
    sys.exit(app.exec_())


