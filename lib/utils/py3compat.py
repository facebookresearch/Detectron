import sys

def bytes2string(x):
    if isinstance(x,bytes):
        return x.decode('ascii')
    if sys.version_info.major == 2 and isinstance(x,unicode): # 2to3 turns "unicode" into "str" but we don't want to decode "str" in python 3
        return x.decode('latin1')  # the pickles from the Model Zoo (as of January 2018) seem to be encoded with latin1; see also https://github.com/tflearn/tflearn/issues/57
    assert isinstance(x,str), str(type(x))
    return x
