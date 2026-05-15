# Dataset package - lazy imports for modules requiring torch
from .property_annotator import PropertyAnnotator

def XRayDataset(*args, **kwargs):
    from .xray_dataset import XRayDataset as _cls
    return _cls(*args, **kwargs)
