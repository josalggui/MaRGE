"""
Created on Thu June 2 2022
@author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
@email: josalggui@i3m.upv.es
@Summary: All sequences on the GUI must be here
"""
import inspect
import os
import importlib

"""
Definition of default sequences
"""

# Note for the users: Now the sequences are added automatically to the defaultsequences dictionary.
# To do that, the user should include the parameter 'toMaRGE' as True in the sequence using:
# self.addParameter(string='toMaRGE', val=True)
# This file should not be modified anymore.

def instantiate_sequences():
    folder = 'seq'
    # Dictionary to store class sequences
    defaultsequences = {}

    # List all .py files in the folder
    py_files = [f for f in os.listdir(folder) if f.endswith('.py')]

    # Populate defaulsequences
    for file in py_files:
        # Remove the .py extension to get the module name
        module_name = file[:-3]

        # Dynamically import the module
        module = importlib.import_module(f"{folder}.{module_name}")

        # Find all classes in the module
        classes = inspect.getmembers(module, inspect.isclass)

        # Populate defaultsequences
        for class_name, class_ in classes:
            try:
                if class_().mapVals['toMaRGE']:
                    defaultsequences[class_().mapVals['seqName']] = class_()
                    print(f'{class_().mapVals['seqName']} added to MaRGE')
            except:
                pass

    return defaultsequences

defaultsequences = instantiate_sequences()

# Note for the users: Now the sequences are added automatically to the defaultsequences dictionary.
# To do that, the user should include the parameter 'toMaRGE' as True in the sequence using:
# self.addParameter(string='toMaRGE', val=True)
# This file should not be modified anymore.
