import datetime, time, requests

def get_timestamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')

def check_hex_value(str_val):
    try:
        hexval = int(str_val, 16)
        return True
    except:
        return False

class Enum(object):

    @classmethod
    def keys(cls):
        return [i for i in cls.__dict__.keys() if i[:2] != "__"]

    @classmethod
    def values(cls):
        return [cls.__dict__[i] for i in cls.__dict__ if i[:2] != "__"]
