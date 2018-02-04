
# Summary of changes

* ``grep -r "python2"`` and replace all ``python2`` to just ``python``

* check ``with open(filename, 'r')`` and ``with open(filename, 'w')``...
 if they are opening pickle or other binary files, they need to be opened as ``'wb'`` or ``'rb'``.

* fix ``urllib`` compatibility (``response.info().getheader`` breaks)

* the pickles from the Model Zoo (as of January 2018) seem to be encoded with latin1,
  so use ``pickle.load(f, encoding='latin1')``; see also https://github.com/tflearn/tflearn/issues/57
  However, Python 2 doesn't have an ``encoding`` argument to pickle, so a check is needed
  (try/except or sys version) for compatibility.

* Be careful when processing values of type ``bytes``, they may be intended to be ascii strings
  (as they were in python 2)... e.g. ``if isinstance(somestr,bytes): somestr.decode('ascii')``,
  which is what ``bytes2string`` does in ``lib/utils/py3compat.py``.

* Also some types appear in Python 2 as ``unicode``... need to decode them using ``latin1``.
  2to3 will change ``unicode`` to ``str``, which will leave an error if using ``unicode_str.decode()``,
  so that has to be checked too.

# Git notes

``2to3`` makes too many changes, and the git changelog would be overwhelming.

Instead this repo remains Python2 code, and fixes issues that ``2to3`` won't fix by itself.

# How to use with Python 3

* To convert to python 3, run this from the root directory: ``2to3 -wn .``
* Let ``#!/usr/bin/env python`` map to python 3 (I'd suggest using a python virtual environment)
