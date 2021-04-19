"""
Connection Dialog
@author:    David Schote
@contact:   david.schote@ovgu.de
@version:   2.0 (Beta)
@change:    13/06/2020

@summary:   Popup for connecting/disconnecting server and host

@status:    Works with Magdeburg server
@todo:      Global list with ip's

"""

from PyQt5.QtCore import QRegExp, pyqtSlot
from PyQt5.QtGui import QRegExpValidator
from PyQt5.uic import loadUiType
#from server.communicationmanager import Com
import subprocess  
import time
import sys
sys.path.append('../marcos_extras')

ConnectionDialog_Form, ConnectionDialog_Base = loadUiType('ui/connDialog.ui')


class ConnectionDialog(ConnectionDialog_Base, ConnectionDialog_Form):

    def __init__(self, parent=None):
        super(ConnectionDialog, self).__init__(parent)

        self.setupUi(self)
        self.parent = parent

#        Com.onStatusChanged.connect(self.setConnectionStatusSlot)

        # connect interface signals
        self.conn_btn.clicked.connect(self.connectClientToServer)
#        self.button_disconnectFromServer.clicked.connect(Com.disconnectClient)
        # self.button_removeServerAddress.clicked.connect(self.connectClientToServer)
        # self.button_addServerAddress.clicked.connect(self.connectClientToServer)
#        self.button_disconnectFromServer.setEnabled(False)

        ipValidator = QRegExp(
            '^(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.)'
            '{3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])$')
        self.ip_box.setValidator(QRegExpValidator(ipValidator, self))
        self.ip_box.addItems(['192.168.1.101'])
        # for item in params.hosts: self.ip_box.addItem(item)
        print("connection dialog ready")

    def connectClientToServer(self):
        
        router_ip = self.ip_box.currentText()
        
#        subprocess.Popen('../marcos_extras/marcos_setup.sh %s %s' % (str(router_ip),'rp-122',), shell=True)
#        time.sleep(60)
#        command1 = "killall marcos_server"
#        subprocess.Popen(["ssh", "root@%s" % str(router_ip), command1],
#                        shell=False,
#                        stdout=subprocess.PIPE,
#                        stderr=subprocess.PIPE)
#        
#        # Connect to the server        
#        command2 = "nohup ~/marcos_server &"
#        ssh = subprocess.Popen(["ssh", "root@%s" % str(router_ip), command2],
#                        shell=False,
#                        stdout=subprocess.PIPE,
#                        stderr=subprocess.PIPE)
#        result = ssh.stdout.readlines()
#        if result == []:
#            error = ssh.stderr.readlines()
#            print(error)
#        else:
#            self.setConnectionStatusSlot("Marcos server is running")
#            print(result)
#            self.close()

    def addNewServerAddress(self):
        # TODO: Global list with ip's
        ip = self.ip_box.currentText()
        """
        if not ip in params.hosts:
            self.ip_box.addItem(ip)
        else: return
        params.hosts = [self.ip_box.itemText(i) for i in range(self.ip_box.count())]
        """
        print(ip)

    def removeServerAddress(self):
        # TODO: Global list with ip's
        idx = self.ip_box.currentIndex()
        print(idx)
        

    @pyqtSlot(str)
    def setConnectionStatusSlot(self, status: str = None) -> None:
        """
        Set the connection status
        @param status:  Server connection status
        @return:        None
        """
        self.status_label.setText(status)
        self.parent.status_connection.setText(status)
        print(status)

#        if status == "Connected":
#            self.button_disconnectFromServer.setEnabled(True)
#            self.button_connectToServer.setEnabled(False)
#        elif status == "Unconnected":
#            self.button_disconnectFromServer.setEnabled(False)
#            self.button_connectToServer.setEnabled(True)
#        else:
#            self.button_disconnectFromServer.setEnabled(True)
#            self.button_connectToServer.setEnabled(True)
