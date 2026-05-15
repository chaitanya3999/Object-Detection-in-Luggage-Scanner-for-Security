# Model1 package - lazy imports to avoid requiring torch at import time

def PropertyYOLO(*args, **kwargs):
    from .architecture import PropertyYOLO as _cls
    return _cls(*args, **kwargs)

def MultiTaskLoss(*args, **kwargs):
    from .loss import MultiTaskLoss as _cls
    return _cls(*args, **kwargs)
