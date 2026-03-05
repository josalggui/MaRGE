"""
Created on Thu June 2 2022
@author: J.M. Algarín, MRILab, i3M, CSIC, Valencia
@email: josalggui@i3m.upv.es
@Summary: All sequences on the GUI must be here
"""
import inspect
import os
import importlib
import re

"""
Definition of default sequences
"""

# Note for the users: Now the sequences are added automatically to the defaultsequences dictionary.
# To do that, the user should include the parameter 'toMaRGE' as True in the sequence using:
# self.addParameter(string='toMaRGE', val=True)
# This file should not be modified anymore.

def _sanitize_key_fragment(value):
    """Sanitize text to be filesystem/UI-safe for internal sequence keys."""
    value = str(value).strip().replace(os.sep, '-')
    value = re.sub(r'[^A-Za-z0-9-]+', '-', value)
    value = re.sub(r'-{2,}', '-', value).strip('-')
    return value or "sequence"


def instantiate_sequences():
    # Get the absolute path to this folder (marge/seq)
    folder = os.path.dirname(__file__)

    # Dictionary to store class sequences and UI display labels
    defaultsequences = {}
    sequence_display_names = {}

    # Recursively list all .py files in marge/seq, except __init__.py files
    py_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            # Skip package init and this loader module itself.
            if file.endswith('.py') and file not in {'__init__.py', 'sequences.py'}:
                py_files.append(os.path.join(root, file))
    py_files.sort()

    # Populate defaultsequences
    for py_file in py_files:
        # Build module import path from the file path (supports nested folders)
        rel_path = os.path.relpath(py_file, folder)
        rel_module_name = rel_path[:-3].replace(os.sep, '.')
        module_name = f"marge.seq.{rel_module_name}"
        folder_prefix = os.path.dirname(rel_path).replace(os.sep, '/')

        try:
            # Dynamically import the module using full path
            module = importlib.import_module(module_name)

            # Find all classes in the module
            classes = inspect.getmembers(module, inspect.isclass)

            # Add to defaultsequences only if toMaRGE is True
            for class_name, class_ in classes:
                try:
                    # Ignore classes imported from other modules.
                    if class_.__module__ != module.__name__:
                        continue

                    sequence = class_()
                    if sequence.mapVals.get('toMaRGE'):
                        seq_name = sequence.mapVals['seqName']

                        # Build internal key:
                        # - root modules keep original seq_name for compatibility
                        # - subfolder modules get a sanitized folder prefix to avoid collisions
                        if folder_prefix:
                            folder_key = _sanitize_key_fragment(folder_prefix.replace('/', '-'))
                            seq_key = f"{folder_key}-{_sanitize_key_fragment(seq_name)}"
                        else:
                            seq_key = seq_name

                        # Ensure key uniqueness even after sanitization.
                        base_key = seq_key
                        suffix = 2
                        while seq_key in defaultsequences:
                            seq_key = f"{base_key}-{suffix}"
                            suffix += 1

                        sequence.internal_key = seq_key
                        defaultsequences[seq_key] = sequence
                        if folder_prefix:
                            sequence_display_names[seq_key] = f"{folder_prefix}/{seq_name}"
                        else:
                            sequence_display_names[seq_key] = seq_name

                        print(f"{sequence_display_names[seq_key]} added to MaRGE (key: {seq_key})")
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

if __name__ == "__main__":
    # Simple loader smoke test: print discovered sequences and their display labels.
    print("=== MaRGE sequence loader test ===")
    print(f"Loaded sequences: {len(defaultsequences)}")
    print("")

    ordered_keys = sorted(defaultsequences.keys(), key=lambda key: sequence_display_names.get(key, key).lower())
    print("UI display names (dropdown order):")
    for seq_key in ordered_keys:
        print(f"  {sequence_display_names.get(seq_key, seq_key)}")
    print("")

    print("Display name -> internal key -> class origin:")
    for seq_key in ordered_keys:
        seq = defaultsequences[seq_key]
        display_name = sequence_display_names.get(seq_key, seq_key)
        origin = f"{seq.__class__.__module__}.{seq.__class__.__name__}"
        print(f"  {display_name} -> key='{seq_key}' -> {origin}")
