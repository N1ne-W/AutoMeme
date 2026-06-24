"""Audio slot with state machine: owner-idempotent, fade-in/out, configurable lifecycle."""
import logging
import os
from enum import Enum

logger = logging.getLogger(__name__)

try:
    from src.diagnostics import log_trace
except ImportError:
    def log_trace(*args, **kwargs): pass


class AudioState(Enum):
    IDLE = "idle"
    FADING_IN = "fading_in"
    PLAYING = "playing"
    FADING_OUT = "fading_out"


class AudioSlot:
    """Single audio slot with owner-idempotent state machine.

    States: IDLE -> FADING_IN -> PLAYING -> FADING_OUT -> IDLE

    - Same owner re-trigger during FADING_IN/PLAYING: ignored (idempotent).
    - Different owner during active: old fades out, new starts.
    - follow_image_lifecycle=True: stop() triggers fade-out.
    - follow_image_lifecycle=False: stop() is ignored, audio lives independently.
    - loop_when_image_alive=True: audio loops while playing.
    - loop_when_image_alive=False: audio plays once, then stays silent until stop.
    """

    def __init__(self, project_root: str):
        self._project_root = project_root
        self._owner_id: str | None = None
        self._state = AudioState.IDLE
        self._volume: float = 0.0
        self._max_volume: float = 0.8
        self._fade_in_ms: int = 300
        self._fade_out_ms: int = 300
        self._follow_image: bool = True
        self._loop: bool = True
        self._sound = None       # pygame.mixer.Sound
        self._channel = None     # pygame.mixer.Channel

    # ?? public API ??

    @property
    def owner_id(self) -> str | None:
        return self._owner_id

    @property
    def state(self) -> AudioState:
        return self._state

    @property
    def volume(self) -> float:
        return self._volume

    def play(self, owner_id: str, audio_rel_path: str,
             follow_image: bool = True, loop: bool = True,
             max_volume: float = 0.8, fade_in_ms: int = 300,
             fade_out_ms: int = 300) -> None:
        """Request audio playback for owner_id.

        Idempotent: if same owner is already FADING_IN or PLAYING, ignore.
        If same owner is FADING_OUT, resume fading in.
        If different owner: fade out old, then start new.
        """
        # Idempotent: same owner already playing
        if self._owner_id == owner_id and self._state in (AudioState.FADING_IN, AudioState.PLAYING):
            logger.debug("[audio] play ignored: owner=%s already active (state=%s)",
                         owner_id, self._state.value)
            log_trace("audio", "audio.slot", "play_ignored_duplicate",
                      owner_id=owner_id, state=self._state.value)
            return

        # Same owner fading out: resume
        if self._owner_id == owner_id and self._state == AudioState.FADING_OUT:
            logger.debug("[audio] play resuming from fade-out: owner=%s", owner_id)
            self._state = AudioState.FADING_IN
            self._fade_in_ms = fade_in_ms
            self._max_volume = max_volume
            self._follow_image = follow_image
            self._loop = loop
            log_trace("audio", "audio.slot", "play_resumed", owner_id=owner_id)
            return

        # Different owner: stop old first
        if self._owner_id is not None and self._owner_id != owner_id:
            logger.debug("[audio] owner replaced: %s -> %s", self._owner_id, owner_id)
            self._hard_stop()

        # Start new
        self._owner_id = owner_id
        self._state = AudioState.FADING_IN
        self._volume = 0.0
        self._max_volume = max_volume
        self._fade_in_ms = max(1, fade_in_ms)
        self._fade_out_ms = max(1, fade_out_ms)
        self._follow_image = follow_image
        self._loop = loop

        self._load_and_play(audio_rel_path)
        logger.info("[audio] started: owner=%s path=%s loop=%s follow=%s",
                    owner_id, audio_rel_path, loop, follow_image)
        log_trace("audio", "audio.slot", "play_started",
                  owner_id=owner_id, path=audio_rel_path,
                  loop=str(loop), follow=str(follow_image))

    def stop(self, owner_id: str) -> None:
        """Request audio stop for owner_id.

        If follow_image_lifecycle=False, stop is ignored (audio lives independently).
        Owner mismatch: ignored.
        If FADING_IN or PLAYING: transition to FADING_OUT.
        """
        if self._owner_id != owner_id:
            logger.debug("[audio] stop ignored: owner mismatch (expected=%s, got=%s)",
                         self._owner_id, owner_id)
            log_trace("audio", "audio.slot", "stop_owner_mismatch",
                      expected_owner=self._owner_id or "none", received_owner=owner_id)
            return

        if not self._follow_image:
            logger.debug("[audio] stop ignored: follow_image_lifecycle=False, owner=%s", owner_id)
            log_trace("audio", "audio.slot", "stop_ignored_independent",
                      owner_id=owner_id)
            return

        if self._state in (AudioState.FADING_IN, AudioState.PLAYING):
            logger.debug("[audio] fade-out started: owner=%s", owner_id)
            self._state = AudioState.FADING_OUT
            log_trace("audio", "audio.slot", "fade_out_started", owner_id=owner_id)

    def tick(self, dt_ms: float) -> None:
        """Per-frame volume update. Call once per frame."""
        if self._state == AudioState.IDLE:
            return
        if self._channel is None:
            return

        if self._state == AudioState.FADING_IN:
            step = (self._max_volume / self._fade_in_ms) * dt_ms
            self._volume = min(self._max_volume, self._volume + step)
            self._apply_volume()
            if self._volume >= self._max_volume:
                self._state = AudioState.PLAYING
                logger.debug("[audio] fade-in complete: owner=%s vol=%.2f",
                             self._owner_id, self._volume)
                log_trace("audio", "audio.slot", "fade_in_complete",
                          owner_id=self._owner_id or "none")

        elif self._state == AudioState.PLAYING:
            # Check natural end only if not looping
            if not self._loop and not self._channel.get_busy():
                logger.debug("[audio] natural end (no loop): owner=%s", self._owner_id)
                log_trace("audio", "audio.slot", "ended_naturally",
                          owner_id=self._owner_id or "none")
                # Stay PLAYING but silent; stop() from image will trigger fade-out

        elif self._state == AudioState.FADING_OUT:
            step = (self._max_volume / self._fade_out_ms) * dt_ms
            self._volume = max(0.0, self._volume - step)
            self._apply_volume()
            if self._volume <= 0.0:
                self._hard_stop()
                logger.debug("[audio] fade-out complete: owner=%s", self._owner_id)
                log_trace("audio", "audio.slot", "fade_out_complete",
                          owner_id=self._owner_id or "none")

    def shutdown(self) -> None:
        """Hard stop and reset on application exit."""
        self._hard_stop()

    # ?? internal ??

    def _load_and_play(self, rel_path: str) -> None:
        import pygame
        full = os.path.join(self._project_root, "assets", rel_path)
        if not os.path.exists(full):
            logger.warning("[audio] file not found: %s", full)
            log_trace("audio", "audio.slot", "asset_missing", path=rel_path)
            self._state = AudioState.IDLE
            return
        try:
            self._sound = pygame.mixer.Sound(full)
            loops = -1 if self._loop else 0
            self._channel = self._sound.play(loops=loops, fade_ms=0)
            self._channel.set_volume(0.0)
        except Exception as e:
            logger.error("[audio] failed to play %s: %s", rel_path, e)
            log_trace("audio", "audio.slot", "play_failed",
                      path=rel_path, error=str(e))
            self._state = AudioState.IDLE

    def _apply_volume(self) -> None:
        if self._channel is not None:
            try:
                self._channel.set_volume(self._volume)
            except Exception:
                pass

    def _hard_stop(self) -> None:
        if self._channel is not None:
            try:
                self._channel.stop()
            except Exception:
                pass
        self._sound = None
        self._channel = None
        self._owner_id = None
        self._state = AudioState.IDLE
        self._volume = 0.0


__all__ = ["AudioSlot", "AudioState"]

