
# Summary of changes

* ``grep -r "python2"`` and replace all ``python2`` to just ``python``

* check ``with open(filename, 'r')`` and ``with open(filename, 'w')``...
 if they are opening pickle or other binary files, they need to be opened as ``'wb'`` or ``'rb'``.

I am intentionally not running ``2to3``, because it makes too many changes, and the git changelog would be overwhelming.

# How to use under Python 3

* Finally, to convert to python 3, run this: ``2to3 -wn .``

