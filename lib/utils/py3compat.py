
def bytes2string(x):
    if isinstance(x,bytes):
        return x.decode('ascii')
    assert isinstance(x,str), str(type(x))
    return x
