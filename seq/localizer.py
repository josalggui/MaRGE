"""
Created on Thu June 2 2022
@author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
@email: josalggui@i3m.upv.es
@Summary: All sequences on the GUI must be here
"""

import os
import sys
# *****************************************************************************
# Add path to the working directory
path = os.path.realpath(__file__)
ii = 0
for char in path:
    if (char == '\\' or char == '/') and path[ii + 1:ii + 14] == 'PhysioMRI_GUI':
        sys.path.append(path[0:ii + 1] + 'PhysioMRI_GUI')
        sys.path.append(path[0:ii + 1] + 'marcos_client')
    ii += 1
# ******************************************************************************
import seq.rare as rare

class Localizer(rare.RARE):
    def __init__(self):
        super(Localizer, self).__init__()
        self.addParameter(key='planes', string='Planes (sag, cor, tra)', val=[1, 1, 1], field='OTH')
        self.planes = self.mapVals['planes']

        self.mapVals['seqName'] = 'Localizer'
        self.mapNmspc['seqName'] = 'LocalizerInfo'
        self.pos = [0, 0, 0]                    # Global position of each axis

if __name__ == '__main__':
    seq = Localizer()
    seq.sequenceRun()
    seq.sequenceAnalysis(obj='Standalone')
