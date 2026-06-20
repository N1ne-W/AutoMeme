import pytest, os
from src.engine.mapping_engine import MappingEngine

FIX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")


class TestMappingEngine:
    def test_loads_mappings(self):
        me = MappingEngine()
        me.load(os.path.join(FIX, "mappings.default.json"), os.path.join(FIX, "mappings.user.json"))
        assert len(me._mappings) >= 2

    def test_match_all_features(self):
        me = MappingEngine()
        me.load(os.path.join(FIX, "mappings.default.json"))
        result = me.match_vision({"is_donk": True})
        assert result is not None and result.id == "donk"

    def test_partial_features_no_match(self):
        me = MappingEngine()
        me.load(os.path.join(FIX, "mappings.default.json"))
        assert me.match_vision({"is_donk": False}) is None

    def test_no_features_no_match(self):
        me = MappingEngine()
        me.load(os.path.join(FIX, "mappings.default.json"))
        assert me.match_vision({}) is None

    def test_heart_mapping(self):
        me = MappingEngine()
        me.load(os.path.join(FIX, "mappings.default.json"))
        r = me.match_vision({"is_omg": True})
        assert r is not None and r.id == "omg"

    def test_priority_desc_sorted(self):
        me = MappingEngine()
        me.load(os.path.join(FIX, "mappings.default.json"))
        priorities = [m.priority for m in me._mappings]
        assert priorities == sorted(priorities, reverse=True)
