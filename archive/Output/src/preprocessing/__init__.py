# Preprocessing package
# Lazy imports to avoid requiring all deps at import time

def preprocess_xray(*args, **kwargs):
    from .pipeline import preprocess_xray as _fn
    return _fn(*args, **kwargs)

def PreprocessingPipeline(*args, **kwargs):
    from .pipeline import PreprocessingPipeline as _cls
    return _cls(*args, **kwargs)

def get_train_augmentation(*args, **kwargs):
    from .augmentation import get_train_augmentation as _fn
    return _fn(*args, **kwargs)

def get_val_augmentation(*args, **kwargs):
    from .augmentation import get_val_augmentation as _fn
    return _fn(*args, **kwargs)
