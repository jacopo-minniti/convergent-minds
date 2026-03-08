from .model import glove

def get_glove_840b():
    return glove('glove.840B.300d', dimensions=300)
