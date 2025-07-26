"""
Created on Thu June 2 2022
@author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
@email: josalggui@i3m.upv.es
@Summary: All sequences on the GUI must be here
"""

import os
import sys
#*****************************************************************************
# Get the directory of the current script
main_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(main_directory)
parent_directory = os.path.dirname(parent_directory)

# Define the subdirectories you want to add to sys.path
subdirs = ['MaRGE', 'marcos_client']

# Add the subdirectories to sys.path
for subdir in subdirs:
    full_path = os.path.join(parent_directory, subdir)
    sys.path.append(full_path)
#******************************************************************************
import marge.seq.rare as rare

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
