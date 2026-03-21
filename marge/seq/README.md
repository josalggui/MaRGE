# Sequence Folder

`marge/seq` contains the MRI sequence modules that can be shown in the MaRGE GUI.

## How loading works

- MaRGE scans `marge/seq` recursively.
- Python files in subfolders are loaded too.
- A sequence is added to the GUI only if its class sets `toMaRGE=True`.
- The GUI list shows root sequences first, then sequences found in subfolders.

## Subfolder support

You can group custom sequences inside subfolders, for example:

```text
marge/seq/custom/autoTuningCustom.py
```

If that file defines a sequence with:

```python
self.addParameter(key='seqName', string='AutoTuningInfo', val='AutoTuningCustom')
self.addParameter(key='toMaRGE', val=True)
```

then the sequence will appear in the GUI as:

```text
custom/AutoTuningCustom
```

## Notes

- Subfolders do not need their own `__init__.py`.
- Keep `seqName` unique across all loaded sequences.
- If two sequences use the same `seqName`, the loader keeps the first one found and prints a warning.
