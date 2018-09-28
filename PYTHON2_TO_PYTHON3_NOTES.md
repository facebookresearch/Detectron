
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

* ``file`` is no longer a keyword in python3; opened text files appear as ``io.TextIOWrapper`` type.

# Git notes

``2to3`` makes too many changes, and the git changelog would be overwhelming.

Instead this repo remains Python2 code, and fixes issues that ``2to3`` won't fix by itself.

This is tested on Ubuntu 16.04 with Python 2.7 and 3.5; it should work with ``tools/infer_simple.py`` and the tests in ``tests_to_pass.sh``.

# How to use with Python 3

* To convert to python 3, run this from the root directory: ``./python2_to_python3_conversion_automated.sh``
* Let ``#!/usr/bin/env python`` map to python 3 (maybe a python virtual environment, bash alias, or Docker)
