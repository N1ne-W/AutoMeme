import pytest, time
from src.engine.cooldown import Cooldown


class TestCooldown:
    def test_initial_not_active(self):
        c = Cooldown(default_cooldown_ms=3000)
        assert not c.is_active

    def test_active_after_trigger(self):
        c = Cooldown(default_cooldown_ms=500)
        c.start(0)
        assert c.is_active

    def test_expires_after_time(self):
        c = Cooldown(default_cooldown_ms=50)
        c.start(0)
        time.sleep(0.1)
        assert not c.is_active

    def test_reset_clears_cooldown(self):
        c = Cooldown(default_cooldown_ms=5000)
        c.start(0)
        assert c.is_active
        c.reset()
        assert not c.is_active

    def test_re_trigger_extends(self):
        c = Cooldown(default_cooldown_ms=5000)
        c.start(0)
        time.sleep(0.1)
        c.start(0)
        assert c.is_active
        assert c.remaining_ms > 4500
