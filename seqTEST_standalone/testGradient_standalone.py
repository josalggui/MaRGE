"""
@author: T. Guallart Naval
MRILAB @ I3M
"""
import os
import sys
#*****************************************************************************
# Add path to the working directory
path = os.path.realpath(__file__)
ii = 0
for char in path:
    if (char=='\\' or char=='/') and path[ii+1:ii+14]=='PhysioMRI_GUI':
        sys.path.append(path[0:ii+1]+'PhysioMRI_GUI')
        sys.path.append(path[0:ii+1]+'marcos_client')
    ii += 1
#******************************************************************************
import experiment as ex
import numpy as np
import matplotlib.pyplot as plt
import configs.hw_config as hw

def testGradient_standalone(
        larmorFreq = 3.060,
        bw = 0.1,

        gAxis = 2,
        gFlattopTime=6.0,
        gRiseTime = 0.5, #ms
        gSteps =10, # El minimo es 2
        gAmp = 0.1, # balanced; 10 V/a.u. Ocra1 conversion factor; [0.1 a.u. 10 A]
        nRepetition = 1,
        extraTime = 100,
        plotSeq = 0,
        init_gpa = False,):

    # Initialize the experiment

    samplingPeriod = 1 / bw  # us
    expt = ex.Experiment(lo_freq=larmorFreq, rx_t=samplingPeriod, init_gpa=init_gpa, gpa_fhdo_offset_time=0)
    samplingPeriod = expt.get_rx_ts()[0]
    # bw = 1 / samplingPeriod  / hw.oversamplingFactor  # MHz
    tStart = 25
    gFlattopTime = gFlattopTime * 1e3
    gRiseTime = gRiseTime * 1e3
    extraTime = extraTime * 1e3
    shimming = [0.0, 0.0, 0.0]

    tUp = np.linspace(tStart, tStart + gRiseTime, num=gSteps, endpoint=False)
    tDown = tUp + gRiseTime + gFlattopTime
    t = np.concatenate((tUp, tDown), axis=0)
    dAmp = gAmp / (gSteps+1)
    aUp = np.linspace(0, gAmp, gSteps , endpoint=True)
    aDown = np.linspace(gAmp, 0, gSteps , endpoint = True)
    a = np.concatenate((aUp, aDown), axis=0)
    if gAxis == 0:
        expt.add_flodict({'grad_vx': (t, a + shimming[0])})
    elif gAxis == 1:
        expt.add_flodict({'grad_vy': (t, a + shimming[1])})
    elif gAxis == 2:
        expt.add_flodict({'grad_vz': (t, a + shimming[2])})

    tEnd = 2*gRiseTime + gFlattopTime + extraTime
    expt.add_flodict({
        'grad_vx': (np.array([tEnd]), np.array([0])),
        'grad_vy': (np.array([tEnd]), np.array([0])),
        'grad_vz': (np.array([tEnd]), np.array([0])),
        'rx0_en': (np.array([tEnd]), np.array([0])),
        'rx1_en': (np.array([tEnd]), np.array([0])),
        'rx_gate': (np.array([tEnd]), np.array([0])),
        'tx0': (np.array([tEnd]), np.array([0 * np.exp(0)])),
        'tx1': (np.array([tEnd]), np.array([0 * np.exp(0)])),
        'tx_gate': (np.array([tEnd]), np.array([0]))
    })
    if plotSeq == 0:
        # Run the experiment and get data
        for i in range(nRepetition):
            text = str(i+1) + ' Running...'
            print(text)
            rxd, msgs = expt.run()
            print(msgs)
        print('End.')
    elif plotSeq == 1:
        expt.plot_sequence()
        plt.show()
    expt.__del__()

#  MAIN  ######################################################################################################
if __name__ == "__main__":
    testGradient_standalone()