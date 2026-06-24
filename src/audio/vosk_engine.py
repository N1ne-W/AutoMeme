"""Vosk 离线语音识别引擎（后台线程 + 完整诊断）。"""
import json
import logging
import os
import queue
import threading
import time

import numpy as np
import sounddevice as sd

from src.diagnostics import new_trace_id, log_trace, Throttle

logger = logging.getLogger(__name__)


class AudioSignal:
    """音频管道输出信号。"""
    def __init__(self, keyword: str, confidence: float, timestamp: float, text: str = "", trace_id: str = ""):
        self.keyword = keyword
        self.confidence = confidence
        self.timestamp = timestamp
        self.text = text
        self.trace_id = trace_id


class VoskEngine:
    """Vosk 语音识别引擎。

    用法:
        engine = VoskEngine(model_path, keywords=["hello"])
        engine.start()
        signal = engine.poll()
        engine.stop()
    """

    def __init__(self, model_path: str, keywords: list[str], sample_rate: int = 16000):
        self._model_path = model_path
        self._keywords = [kw.lower() for kw in keywords]
        self._sample_rate = sample_rate
        self._queue: queue.Queue[AudioSignal] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._running = False
        self._available = False
        self._rms_throttle = Throttle()
        self._last_error: str | None = None

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def keywords(self) -> list[str]:
        return self._keywords

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def start(self) -> bool:
        """启动后台识别线程。返回是否成功。"""
        if not self._keywords:
            logger.info("[voice] SKIPPED: no keywords configured")
            log_trace("init", "voice.keyword", "keyword_rules_invalid", reason="empty_keywords")
            self._last_error = "no_keywords"
            return False

        if not os.path.exists(self._model_path):
            logger.warning("[voice] SKIPPED: model not found at %s", self._model_path)
            log_trace("init", "voice.vosk", "model_missing", path=self._model_path)
            self._last_error = "model_missing"
            return False

        try:
            import vosk
            vosk.SetLogLevel(-1)
            self._model = vosk.Model(self._model_path)
            self._recognizer = vosk.KaldiRecognizer(self._model, self._sample_rate)
            self._recognizer.SetWords(True)
            self._recognizer.SetPartialWords(True)
            log_trace("init", "voice.vosk", "model_loaded",
                      path=self._model_path, sample_rate=str(self._sample_rate))
        except Exception as e:
            logger.error("[voice] Vosk init failed: %s", e)
            log_trace("init", "voice.vosk", "recognizer_error", error=str(e))
            self._last_error = f"vosk_init: {e}"
            self._available = False
            return False

        self._running = True
        self._available = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("[voice] Started: %d keywords, model=%s", len(self._keywords), self._model_path)
        log_trace("init", "voice.thread", "recognizer_thread_started",
                  keywords=str(len(self._keywords)), sample_rate=str(self._sample_rate))
        return True

    def poll(self) -> AudioSignal | None:
        """非阻塞获取最新识别信号。"""
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._available = False
        logger.info("[voice] Stopped")
        log_trace("cleanup", "voice.thread", "recognizer_thread_stopped")

    def _run(self) -> None:
        """后台线程：采音并识别。"""
        try:
            logger.debug("[voice] Audio stream opening: rate=%d, channels=1", self._sample_rate)
            log_trace("init", "voice.input", "audio_stream_started",
                      sample_rate=str(self._sample_rate), device="default")

            with sd.InputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype="int16",
                callback=self._audio_callback,
            ):
                while self._running:
                    sd.sleep(100)
            log_trace("cleanup", "voice.input", "audio_stream_stopped")
        except Exception as e:
            logger.error("[voice] Audio stream crashed: %s", e)
            log_trace("error", "voice.input", "audio_device_error", error=str(e))
            log_trace("error", "voice.thread", "recognizer_thread_crashed", error=str(e))
            self._last_error = f"audio_stream: {e}"
            self._available = False

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        """音频回调：送 Vosk 识别。"""
        if status:
            logger.warning("[voice] Audio status: %s", status)
            log_trace("warn", "voice.input", "audio_device_error", status=str(status))

        # RMS 音量节流输出
        rms = float(np.sqrt(np.mean(np.square(indata.astype(np.float32)))))
        interval_ms = 1000
        try:
            from src.diagnostics import _debug_cfg
            interval_ms = _debug_cfg.get("audio_level_interval_ms", 1000)
        except Exception:
            pass
        if self._rms_throttle.should_log(f"rms_{interval_ms}"):
            rms_threshold = 500
            logger.debug("[voice] Mic RMS: %.0f %s", rms, "(silent)" if rms < rms_threshold else "")
            log_trace("audio", "voice.input", "audio_level", rms=f"{rms:.0f}",
                      status="silent" if rms < rms_threshold else "active")

        data = indata.tobytes()

        if self._recognizer.AcceptWaveform(data):
            result = json.loads(self._recognizer.Result())
            text = result.get("text", "").strip()
            if text:
                trace_id = new_trace_id()
                logger.info("[voice] Final text: '%s'", text)
                log_trace(trace_id, "voice.vosk", "speech_final", text=text)
                self._check_keywords(text.lower(), time.time(), trace_id)
        else:
            partial = json.loads(self._recognizer.PartialResult())
            partial_text = partial.get("partial", "").strip()
            if partial_text:
                try:
                    from src.diagnostics import _debug_cfg
                    if _debug_cfg.get("verbose_partial_speech", False):
                        logger.debug("[voice] Partial: '%s'", partial_text)
                        log_trace("audio", "voice.vosk", "speech_partial", text=partial_text)
                except Exception:
                    pass

    def _check_keywords(self, text: str, timestamp: float, trace_id: str) -> None:
        """检查文本中是否包含关键词。"""
        matched = None
        for kw in self._keywords:
            if kw.isascii():
                if re.search(r"\b" + re.escape(kw) + r"\b", text):
                    matched = kw
                    break
            else:
                if kw in text:
                    matched = kw
                    break

        if matched:
            conf = 1.0 if text == matched else 0.8
            signal = AudioSignal(keyword=matched, confidence=conf, timestamp=timestamp,
                                 text=text, trace_id=trace_id)
            try:
                self._queue.put_nowait(signal)
            except queue.Full:
                pass
            log_trace(trace_id, "voice.keyword", "keyword_matched",
                      keyword=matched, text=text, confidence=f"{conf:.2f}")
        else:
            log_trace(trace_id, "voice.keyword", "keyword_missed",
                      text=text, rules=str(len(self._keywords)))
