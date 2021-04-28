"""
Connection 
@author:    
@contact:   
@version:   
@change:    

@summary:  


"""
from PyQt5.QtCore import pyqtSlot
import subprocess  
import sys
sys.path.append('../marcos_extras')
sys.path.append('../marcos_client')

class ServerConnection():

    def __init__(self, parent=None, ip=None):
        super(ServerConnection, self).__init__(parent)

        # Make parent reachable from outside __init__
        self.parent = parent
        self.ip = ip
        
    def connectClientToServer(self):

#        
#        subprocess.Popen('../marcos_extras/marcos_setup.sh %s %s' % (str(router_ip),'rp-122',), shell=True)
#        time.sleep(60)
#        command1 = "killall marcos_server"
#        subprocess.Popen(["ssh", "root@%s" % str(router_ip), command1],
#                        shell=False,
#                        stdout=subprocess.PIPE,
#                        stderr=subprocess.PIPE)
#        
        # Connect to the server        
        command = "~/marcos_server &"
        ssh = subprocess.Popen(["ssh", "root@%s" % self.ip, command],
                        shell=False,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)
        result = ssh.stdout.readlines()
        print(result)
        if result == []:
            error = ssh.stderr.readlines()
            print(error)
        else:
            self.setConnectionStatusSlot("Marcos server is running")
            print(result)
            self.close()

    @pyqtSlot(str)
    def setConnectionStatusSlot(self, status: str = None) -> None:
        """
        Set the connection status
        @param status:  Server connection status
        @return:        None
        """
        self.parent.status_connection.setText(status)
        print(status)


