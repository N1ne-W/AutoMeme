# Lazy imports to avoid eager cv2 import
def __getattr__(name):
    if name == "Camera":
        from .camera import Camera
        return Camera
    if name == "HolisticRunner":
        from .holistic_runner import HolisticRunner
        return HolisticRunner
    if name == "FeatureExtractor":
        from .feature_extractor import FeatureExtractor
        return FeatureExtractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
