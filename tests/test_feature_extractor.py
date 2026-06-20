import pytest, os
from src.vision.feature_extractor import FeatureExtractor

FIX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")


class TestFeatureExtractor:
    def test_loads_features(self):
        fe = FeatureExtractor(os.path.join(FIX, "features.json"))
        assert "is_squat" in fe.feature_names
        assert "is_victory" in fe.feature_names
        assert "is_heart" in fe.feature_names

    def test_no_person_all_false(self):
        fe = FeatureExtractor(os.path.join(FIX, "features.json"))
        fv = fe.extract(None, 0, 0.0)
        for name in fe.feature_names:
            assert fv.features[name] is False

    def test_metadata(self):
        fe = FeatureExtractor(os.path.join(FIX, "features.json"))
        fv = fe.extract(None, 42, 1.5)
        assert fv.frame_id == 42
        assert fv.timestamp == 1.5

    def test_missing_config_empty(self):
        fe = FeatureExtractor("/nonexistent/path.json")
        assert fe.feature_names == []
        assert fe.extract(None, 0, 0.0).features == {}
