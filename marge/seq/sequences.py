"""
Created on Thu June 2 2022
@author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
@email: josalggui@i3m.upv.es
@Summary: All sequences on the GUI must be here
"""
import inspect
import os
import importlib
import importlib.util

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

    # Store the sequence instances and the names shown in the GUI.
    defaultsequences = {}
    sequence_display_names = {}

    # Search recursively so custom sequences inside subfolders are discovered too.
    py_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith('.py') and file not in {'__init__.py', 'sequences.py'}:
                py_files.append(os.path.join(root, file))
    py_files.sort()

    # Populate defaultsequences
    for py_file in py_files:
        rel_path = os.path.relpath(py_file, folder)
        rel_module_name = rel_path[:-3].replace(os.sep, '.')
        folder_prefix = os.path.dirname(rel_path).replace(os.sep, '/')
        module_name = f"marge.seq.{rel_module_name}"

        try:
            # Load modules from file paths so subfolders do not need to be Python packages.
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            if spec is None or spec.loader is None:
                raise ImportError(f"Unable to create import spec for {py_file}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find all classes in the module
            classes = inspect.getmembers(module, inspect.isclass)

            # Add to defaultsequences only if toMaRGE is True
            for class_name, class_ in classes:
                try:
                    # Ignore classes re-exported from imports inside the module.
                    if class_.__module__ != module.__name__:
                        continue

                    sequence = class_()
                    if sequence.mapVals['toMaRGE']:
                        seq_name = sequence.mapVals['seqName']
                        if seq_name in defaultsequences:
                            print(f"WARNING: Duplicate sequence name '{seq_name}' found in {rel_path}.")
                            continue

                        defaultsequences[seq_name] = sequence
                        if folder_prefix:
                            sequence_display_names[seq_name] = f"{folder_prefix}/{seq_name}"
                        else:
                            sequence_display_names[seq_name] = seq_name
                        print(f"{sequence_display_names[seq_name]} added to MaRGE")
                except:
                    pass
        except Exception as e:
            print(f"Error importing module {module_name}: {e}")

    return defaultsequences, sequence_display_names

defaultsequences, sequence_display_names = instantiate_sequences()

# Note for the users: Now the sequences are added automatically to the defaultsequences dictionary.
# To do that, the user should include the parameter 'toMaRGE' as True in the sequence using:
# self.addParameter(string='toMaRGE', val=True)
# This file should not be modified anymore.
