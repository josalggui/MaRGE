"""
Communication Manager

@author:    David Schote
@contact:   david.schote@ovgu.de
@version:   1.0
@change:    02/05/2020

@summary:   Manages the connection to the server, constructs and sends packages (via msgpack)

@status:    Under testing
@todo:

"""

from PyQt5.QtNetwork import QAbstractSocket, QTcpSocket
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QObject
from sequencesnamespace import Namespace as nmspc
from warnings import warn
import numpy as np
import struct
import msgpack


states = {
    QAbstractSocket.UnconnectedState: "Unconnected",
    QAbstractSocket.HostLookupState: "Host Lookup",
    QAbstractSocket.ConnectingState: "Connecting",
    QAbstractSocket.ConnectedState: "Connected",
    QAbstractSocket.BoundState: "Bound",
    QAbstractSocket.ClosingState: "Closing Connection",
}
status = QAbstractSocket.SocketState

fpga_clk = 125.0

class Commands:
    """
    Commands Class for Marcos-Server
    """
    fpgaClock = 'fpga_clk' # array of 3 values unsigned int (clock words)
    localOscillatorFrequency = 'lo_freq' # unsigned int (local oscillator freq. for TX/RX)
    txClockDivider = 'tx_div' # unsigned int (clock divider for RF TX samples)
    rfAmplitude = 'rf_amp' # unsigned int 16 bit (RF amplitude)
    rxRate = 'rx_rate' # unsigned int 16 bit (tx sample rate???)
    txSampleSize = 'tx_size' # unsigned int 16 bit (number of TX samples to return)
    txSamplesPerPulse = 'tx_samples' # unsigned int (number of TX samples per pulse)
    gradientOffsetX = 'grad_offs_x' # unsigned int (X gradient channel shim)
    gradientOffsetY = 'grad_offs_y' # unsigned int (Y gradient channel shim)
    gradientOffsetZ = 'grad_offs_z' # unsigned int (Z gradient channel shim)
    gradientMemoryX = 'grad_mem_x' # binary byte array (write X gradient channel memory)
    gradientMemoryY = 'grad_mem_y' # binary byte array (write Y gradient channel memory)
    gradientMemoryZ = 'grad_mem_z' # binary byte array (write Z gradient channel memory)
    recomputeTxPulses = 'recomp_pul' # boolean (recompute the TX pulses)
    txRfWaveform = 'raw_tx_data' # binary byte array (write the RF waveform)
    sequenceData = 'seq_data' # binary byte array (pulse sequence instructions)
    runAcquisition = 'acq' # unsigned int [samples] (runs 'seq_data' and returns array of 64-bit complex floats, length = samples)
    testRxThroughput = 'test_throughput' # unsigned int [arg] (return array map, array-length = arg)
    requestPacket = 0

# TODO: Create a config file for the following
fpga_clk_frequency_MHz = 125


class CommunicationManager(QTcpSocket, QObject):
    """
    Communication Manager Class
    """
    statusChanged = pyqtSignal(str, name='onStatusChanged')

    def __init__(self):
        super(CommunicationManager, self).__init__()

        self.stateChanged.connect(self.getConnectionStatus)

    def connectClient(self, ip: str) -> [bool, int]:
        """
        Connect server and host through server's ip
        @param ip:  IP address of the server
        @return:    success of connection
        """
        self.connectToHost(ip, 11111)
        self.waitForConnected(2000)
        if self.state() == QAbstractSocket.ConnectedState:
            print("Connection to server established.")
            return True
        else:
            print("Connection to server failed.")
            return False

    def disconnectClient(self) -> bool:
        """
        Disconnects server and host
        @return:    success of disconnection
        """
        self.disconnectFromHost()
        if self.state() is QAbstractSocket.UnconnectedState:
            print("Disconnected from server.")
            return True
        else:
            return False

    @pyqtSlot(status)
    def getConnectionStatus(self, state: status = None) -> None:
        if state in states:
            self.statusChanged.emit(states[state])
        else:
            self.statusChanged.emit(str(state))

    def waitForTransmission(self) -> None:
        """
        Wait until bytes are written on server
        @return:    None
        """
        while True:
            if not self.waitForBytesWritten():
                break

    @staticmethod
    def constructPropertyPacket(operation) -> dict:

        packet: dict = {}

        if hasattr(operation, 'systemproperties'):
            sys_prop = operation.systemproperties
            for key in list(sys_prop.keys()):
                if len(sys_prop[key]) == 3:
                    if key == nmspc.frequency:
                        # TODO: Find alternative way for this ?
                        packet[sys_prop[key][2]] = int(np.round(sys_prop[key][0] /
                                                                fpga_clk_frequency_MHz * (1 << 30))) & 0xfffffff0 | 0xf
                        continue
                    packet[sys_prop[key][2]] = sys_prop[key][0]

        # TODO: Integrate gradient offsets in gradient waveform (maybe leave it for spectroscopy)
        """
        if hasattr(operation, 'gradientshims'):
            shim = operation.gradientshims
            packet[Commands.gradientOffsetX] = shim[nmspc.x_grad][0]
            packet[Commands.gradientOffsetY] = shim[nmspc.y_grad][0]
            packet[Commands.gradientOffsetZ] = shim[nmspc.z_grad][0]
        """
        return packet

    @staticmethod
    def constructSequencePacket(operation) -> dict:

        package: dict = {}

        if hasattr(operation, 'pulsesequence') and len(operation.pulsesequence) > 1:
            seq = operation.pulsesequence
            package[Commands.sequenceData] = seq[nmspc.sequence][1]
        else:
            warn("ERROR: No sequence bytestream!")

        return package

    def sendPacket(self, packet):
        self.write(msgpack.packb(packet))
        unpacker = msgpack.Unpacker()
        self.waitForReadyRead(1000)

        while True:
            buf = self.read(100)
            if not buf:
                break
            unpacker.feed(buf)
            for o in unpacker:  # ugly way of doing it
                return o  # quit function after 1st reply (could make this a thread in the future)


    def setFrequency(self, freq: float) -> None:
        # TODO: Reimplement this function for new server
        """
        Set excitation frequency on the server
        @param freq:    Frequency in MHz
        @return:        None
        """
        self.write(struct.pack('<I', 2 << 28 | int(1.0e6 * freq)))
        print("Set frequency!")


Com = CommunicationManager()
