# Utils package
# Import on demand — visualization requires matplotlib
from .config import load_config

def visualize_detections(*args, **kwargs):
    from .visualization import visualize_detections as _fn
    return _fn(*args, **kwargs)

def plot_training_curves(*args, **kwargs):
    from .visualization import plot_training_curves as _fn
    return _fn(*args, **kwargs)
