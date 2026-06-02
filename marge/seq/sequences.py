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
    # Get the absolute path to this folder (marge/seq)
    folder = os.path.dirname(__file__)

    # Dictionary to store class sequences
    defaultsequences = {}

    # List all .py files in the folder, except __init__.py
    py_files = [f for f in os.listdir(folder) if f.endswith('.py') and f != '__init__.py']

    # Populate defaultsequences
    for file in py_files:
        # Remove the .py extension to get the module name
        module_name = file[:-3]

        try:
            # Dynamically import the module using full path
            module = importlib.import_module(f"marge.seq.{module_name}")

            # Find all classes in the module
            classes = inspect.getmembers(module, inspect.isclass)

            # Add to defaultsequences only if toMaRGE is True
            for class_name, class_ in classes:
                try:
                    if class_().mapVals['toMaRGE']:
                        defaultsequences[class_().mapVals['seqName']] = class_()
                        print(f"{class_().mapVals['seqName']} added to MaRGE")
                except:
                    pass
        except Exception as e:
            print(f"Error importing module {module_name}: {e}")

    return defaultsequences

defaultsequences = instantiate_sequences()

# Note for the users: Now the sequences are added automatically to the defaultsequences dictionary.
# To do that, the user should include the parameter 'toMaRGE' as True in the sequence using:
# self.addParameter(string='toMaRGE', val=True)
# This file should not be modified anymore.
