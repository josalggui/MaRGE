# Sequence Loading

`marge/seq` contains sequence modules that can be shown in the MaRGE GUI.

## How It Works

- MaRGE scans `marge/seq` recursively.
- Every `.py` file is checked except `__init__.py` and `sequences.py`.
- Files in subfolders are loaded directly from their path, so subfolders do not need an `__init__.py`.
- A file is only loaded as a sequence module if it defines a class inheriting `MRIBLANKSEQ`, for example `class MySeq(blankSeq.MRIBLANKSEQ):`.
- Imported classes are ignored; only classes defined in the scanned module are considered.
- A sequence is added only if its class instance sets `mapVals['toMaRGE']` to `True`.
- Helper files used to split complex logic are ignored unless they define a `MRIBLANKSEQ` subclass.

## Naming

- The internal lookup key is `mapVals['seqName']`.
- Sequences found in subfolders are shown in the GUI with a folder-prefixed label such as `custom/MySequence`.
- Root-level sequences are shown with just their `seqName`.

## Duplicates

- `seqName` must be unique across all loaded sequences.
- If two sequences use the same `seqName`, the first one found after path sorting is kept.
- Later duplicates are skipped and a warning is printed during loading.

## Example

If you add:

```text
marge/seq/custom/my_sequence.py
```

and that file defines a sequence with:

```python
self.addParameter(key='seqName', val='MySequence')
self.addParameter(key='toMaRGE', val=True)
```

then it will appear in the GUI as:

```text
custom/MySequence
```
