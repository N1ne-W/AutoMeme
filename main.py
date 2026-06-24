"""
AutoMeme - ?????
?? AI ??????????????
License: GNU GPL v3
"""
import os
import sys
import time
import logging
import yaml

# ?????
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

# ????????
os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)

# --- ?? ---
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(PROJECT_ROOT, "logs", "automeme.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

log = logging.getLogger("main")

# --- ?????? ---
def load_app_config() -> dict:
    config_path = os.path.join(PROJECT_ROOT, "config", "app.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    log.warning("app.yaml not found, using defaults")
    return {}

app_config = load_app_config()

# ???????
from src.diagnostics import init_debug_config
init_debug_config(app_config.get("debug", {}))

# --- ???? ---
from src.vision import Camera, HolisticRunner, FeatureExtractor
from src.engine import VisionSignal, StateMachine, MappingEngine
from src.audio import VoskEngine, extract_keywords
from src.engine.signals import AudioSignal as EngineAudioSignal
from src.diagnostics import new_trace_id, log_trace


def main():
    log.info("=== AutoMeme ?? ===")

    camera_cfg = app_config.get("camera", {})
    mp_cfg = app_config.get("mediapipe", {})
    debug_cfg = app_config.get("debug", {})

    # 1. ???????
    camera = Camera(
        index=camera_cfg.get("index", 0),
        width=camera_cfg.get("width", 640),
        height=camera_cfg.get("height", 480),
        fps=camera_cfg.get("fps", 30),
    )
    if not camera.open():
        log.error("?????????")
        return

    model_path = os.path.join(PROJECT_ROOT, mp_cfg.get("model_path", "assets/models/holistic_landmarker.task"))
    holistic = HolisticRunner(
        model_path=model_path,
        min_detection_confidence=mp_cfg.get("min_detection_confidence", 0.5),
        min_tracking_confidence=mp_cfg.get("min_tracking_confidence", 0.5),
    )
    if not holistic.initialize():
        log.error("MediaPipe ????????")
        camera.release()
        return

    feature_extractor = FeatureExtractor(
        os.path.join(PROJECT_ROOT, "config", "features.json")
    )
    log.info("?????: %s", feature_extractor.feature_names)

    # 2. ?????
    mapping_engine = MappingEngine()
    mapping_engine.load(
        default_path=os.path.join(PROJECT_ROOT, "config", "mappings.default.json"),
        user_path=os.path.join(PROJECT_ROOT, "config", "mappings.user.json"),
    )
    state_machine = StateMachine(mapping_engine)

    # Build feature->mapping lookup for hold display tracking
    _feature_to_mapping = {}
    _mapping_lookup = {}
    for m in mapping_engine.mappings:
        _mapping_lookup[m.id] = m
        for f in m.features:
            if f not in _feature_to_mapping:
                _feature_to_mapping[f] = []
            _feature_to_mapping[f].append(m)

    # 2.4 ????????????????/?????
    gesture_cfg = app_config.get("gesture", {})
    from src.engine.gesture_stabilizer import GestureStabilizer
    stabilizer = GestureStabilizer(
        enter_debounce_ms=gesture_cfg.get("enter_debounce_ms", 150),
        exit_debounce_ms=gesture_cfg.get("exit_debounce_ms", 600),
        min_hold_ms=gesture_cfg.get("min_hold_ms", 0),
    )
    log.info("Gesture stabilizer: enter=%dms exit=%dms min_hold=%dms",
             stabilizer.enter_debounce_ms, stabilizer.exit_debounce_ms,
             gesture_cfg.get("min_hold_ms", 0))
    # Apply per-mapping debounce overrides (highest priority wins per feature)
    _feature_debounce_set = set()
    for m in sorted(mapping_engine.mappings, key=lambda x: -x.priority):
        for f in m.features:
            if f in _feature_debounce_set:
                continue
            _feature_debounce_set.add(f)
            if m.enter_debounce_ms > 0 or m.exit_debounce_ms > 0 or m.min_hold_ms > 0:
                stabilizer.set_feature_config(
                    f,
                    enter_ms=m.enter_debounce_ms if m.enter_debounce_ms > 0 else None,
                    exit_ms=m.exit_debounce_ms if m.exit_debounce_ms > 0 else None,
                    min_hold_ms=m.min_hold_ms if m.min_hold_ms > 0 else None,
                )
                log.debug("Stabilizer override for %s: enter=%d exit=%d min_hold=%d",
                          f, m.enter_debounce_ms, m.exit_debounce_ms, m.min_hold_ms)

    # 2.5 ???????
    vosk_cfg = app_config.get("vosk", {})
    model_path = vosk_cfg.get("model_path", "assets/models/vosk-model-small-cn-0.22")
    model_full = os.path.join(PROJECT_ROOT, model_path)

    keywords = extract_keywords(
        os.path.join(PROJECT_ROOT, "config", "mappings.default.json")
    )
    user_kw = extract_keywords(
        os.path.join(PROJECT_ROOT, "config", "mappings.user.json")
    )
    all_keywords = list(set(keywords + user_kw))

    vosk_engine = VoskEngine(
        model_path=model_full,
        keywords=all_keywords,
        sample_rate=vosk_cfg.get("sample_rate", 16000),
    )
    if vosk_engine.start():
        log.info("Vosk engine ready: %d keywords", len(all_keywords))
    else:
        log.info("Vosk engine not available (model missing: %s, keywords: %d)", not os.path.exists(model_full), len(all_keywords))

    # 3. ??? PyGame ??
    import pygame
    import cv2
    import numpy as np

    pygame.init()
    pygame.mixer.init()
    renderer_cfg = app_config.get("renderer", {})
    screen_w = renderer_cfg.get("width", 1280)
    screen_h = renderer_cfg.get("height", 720)
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption("AutoMeme - Debug Mode")
    clock = pygame.time.Clock()

    # ??????
    def load_image(rel_path: str | None):
        if rel_path is None:
            return None
        full = os.path.join(PROJECT_ROOT, "assets", rel_path)
        if not os.path.exists(full):
            return None
        try:
            img = pygame.image.load(full).convert_alpha()
            return img
        except Exception as e:
            log.warning("Failed to load image %s: %s", rel_path, e)
            return None

    # ??? default images
    default_images = {}
    for m in mapping_engine.mappings:
        if m.image_path and m.enabled:
            default_images[m.id] = load_image(m.image_path)

    # --- ??? ---
    frame_id = 0
    active_meme_id: str | None = None
    fade_alpha = 0
    fade_speed = 15
    fading_out_id: str | None = None                # ???????????? ID
    # ?? ????? ??
    audio_cfg = app_config.get("audio", {})
    from src.engine.audio_slot import AudioSlot
    audio_slot = AudioSlot(PROJECT_ROOT)
    running = True

    # ?? Hold + One-shot display ??
    hold_display_id: str | None = None                     # ?? hold ?? (mapping_id)
    hold_owner_feature: str | None = None                   # hold ???? feature_id??? owner ?????
    one_shot_items: list[dict] = []                         # [{mapping_id, start_time, duration_ms, image}]

    log.info("?????")
    while running:
        dt = clock.tick(30) / 1000.0

        # ??
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # ???
        ret, frame = camera.read()
        if not ret:
            break

        # MediaPipe ??
        results = holistic.process(frame)
        now = time.time()

        # ????
        fv = feature_extractor.extract(results, frame_id, now)

        # ??????
        audio_signal = None
        trace_id = None
        if vosk_engine.is_available:
            raw = vosk_engine.poll()
            if raw is not None:
                trace_id = new_trace_id()
                log_trace(trace_id, "voice.keyword", "keyword_matched",
                          keyword=raw.keyword, confidence=f"{raw.confidence:.2f}")
                audio_signal = EngineAudioSignal(
                    keyword=raw.keyword,
                    confidence=raw.confidence,
                    timestamp=raw.timestamp,
                )

        # ??????????????? hold display ???/???
        gesture_events = stabilizer.update(fv.features, now)

        # ?????????? raw features ???????? stabilizer ???
        event = state_machine.update(
            vision_signal=VisionSignal(
                features=fv.features,
                frame_id=frame_id,
                timestamp=now,
            ),
            audio_signal=audio_signal,
            trace_id=trace_id or "",
        )

        # ?? ??????? hold / duration ??
        if event is not None:
            log.info("??! mapping=%s type=%s image=%s mode=%s",
                     event.mapping_id, event.action_type, event.image_path, event.display_mode)
            if trace_id:
                log_trace(trace_id, "state.machine", "trigger_accepted",
                          mapping_id=event.mapping_id, action_type=event.action_type,
                          display_mode=event.display_mode)
            debug_cfg = app_config.get("debug", {})
            if debug_cfg.get("trace_renderer", False):
                log_trace(trace_id or "no_trace", "renderer.pygame", "render_request_received",
                          mapping_id=event.mapping_id, image_path=event.image_path or "null",
                          display_mode=event.display_mode)

            if event.display_mode == "hold":
                # Hold-type: state machine confirmed gesture; stabilizer already tracking it
                old_hold = hold_display_id
                hold_display_id = event.mapping_id
                # Track the primary feature as owner for safe clearing
                mapping = _mapping_lookup.get(event.mapping_id)
                if mapping and mapping.features:
                    hold_owner_feature = mapping.features[0]
                else:
                    hold_owner_feature = event.mapping_id
                log_trace(trace_id or "no_trace", "renderer.pygame", "display_hold_started",
                          mapping_id=event.mapping_id, replaced=old_hold or "none",
                          owner_feature=hold_owner_feature or "unknown")
                # ?? ???hold ????? ??
                if event.action_type in ("both", "audio") and event.audio_path:
                    audio_slot.play(
                        event.mapping_id, event.audio_path,
                        follow_image=audio_cfg.get("follow_image_lifecycle", True),
                        loop=audio_cfg.get("loop_when_image_alive", True),
                        max_volume=audio_cfg.get("master_volume", 0.8),
                        fade_in_ms=audio_cfg.get("fade_in_ms", 300),
                        fade_out_ms=audio_cfg.get("fade_out_ms", 300),
                    )
            else:
                # Duration-type (voice): add to one-shot queue
                one_shot_items.append({
                    "mapping_id": event.mapping_id,
                    "start_time": now,
                    "duration_ms": event.duration_ms,
                    "image": event.image_path,
                })
                log_trace(trace_id or "no_trace", "renderer.pygame", "one_shot_display_started",
                          mapping_id=event.mapping_id, duration_ms=str(event.duration_ms))
                # ?? ???duration ????? ??
                if event.action_type in ("both", "audio") and event.audio_path:
                    audio_slot.play(
                        event.mapping_id, event.audio_path,
                        follow_image=audio_cfg.get("follow_image_lifecycle", True),
                        loop=audio_cfg.get("loop_when_image_alive", True),
                        max_volume=audio_cfg.get("master_volume", 0.8),
                        fade_in_ms=audio_cfg.get("fade_in_ms", 300),
                        fade_out_ms=audio_cfg.get("fade_out_ms", 300),
                    )

        # ?? Stabilizer-driven hold display management ??
        # Process gesture events from the time-based stabilizer
        for ge in gesture_events:
            if ge.event_type == "started":
                # Feature stabilized as active - find its mapping
                feature_mappings = _feature_to_mapping.get(ge.feature_id, [])
                # Priority-sorted (highest first)
                feature_mappings.sort(key=lambda m: -m.priority)
                for fm in feature_mappings:
                    if fm.enabled and fm.display_mode == "hold":
                        old_hold = hold_display_id
                        hold_display_id = fm.id
                        hold_owner_feature = ge.feature_id
                        log.info("[stabilizer] Hold started: mapping=%s feature=%s (replaced=%s)",
                                 fm.id, ge.feature_id, old_hold or "none")
                        log_trace("gesture", "renderer.pygame", "display_hold_started",
                                  mapping_id=fm.id, feature_id=ge.feature_id,
                                  replaced=old_hold or "none")
                        break  # Only one hold at a time

            elif ge.event_type == "ended":
                # Feature ended - only clear if it owns the current hold slot
                if hold_owner_feature is not None and hold_display_id is not None:
                    # Check if this ended feature owns the current display
                    current_mapping = _mapping_lookup.get(hold_display_id)
                    if current_mapping and ge.feature_id in current_mapping.features:
                        ended_mapping_id = hold_display_id
                        log.info("[stabilizer] Hold ended: mapping=%s feature=%s",
                                 hold_display_id, ge.feature_id)
                        log_trace("gesture", "renderer.pygame", "display_hold_ended",
                                  mapping_id=hold_display_id, feature_id=ge.feature_id,
                                  reason="exit_debounce_expired")
                        hold_display_id = None
                        hold_owner_feature = None
                        audio_slot.stop(ended_mapping_id)
                    else:
                        # Owner mismatch - this ended feature doesn''t own current slot
                        log.debug("[stabilizer] Ended feature=%s ignored: current owner=%s (mapping=%s)",
                                  ge.feature_id, hold_owner_feature, hold_display_id)
                        log_trace("gesture", "renderer.pygame", "clear_ignored",
                                  ended_feature=ge.feature_id,
                                  current_owner=hold_owner_feature or "none",
                                  reason="owner_mismatch")

        # ?? Fallback: belt-and-suspenders state check ??
        if hold_display_id and hold_owner_feature:
            if not stabilizer.is_active_or_grace(hold_owner_feature):
                ended_mapping_id = hold_display_id
                log.info("[stabilizer] Hold cleared via state check: mapping=%s feature=%s",
                         hold_display_id, hold_owner_feature)
                log_trace("gesture", "renderer.pygame", "display_hold_ended",
                          mapping_id=hold_display_id, feature_id=hold_owner_feature,
                          reason="inactive_state")
                hold_display_id = None
                hold_owner_feature = None
                audio_slot.stop(ended_mapping_id)

        # ?? One-shot queue: expire old items ??
        one_shot_items = [item for item in one_shot_items
                          if now - item["start_time"] < item["duration_ms"] / 1000.0]

        # ?? Determine current display ??
        # Priority: hold display > highest-priority active one-shot
        prev_active = active_meme_id
        active_meme_id = None
        if hold_display_id:
            active_meme_id = hold_display_id
        elif one_shot_items:
            # Pick highest priority one-shot
            best = None
            best_pri = -1
            for item in one_shot_items:
                m = _mapping_lookup.get(item["mapping_id"])
                pri = m.priority if m else 0
                if pri > best_pri:
                    best_pri = pri
                    best = item["mapping_id"]
            active_meme_id = best

        # ?? Fade with smooth transition (image + audio) ??
        # When display target changes, old image continues rendering during fade-out
        if prev_active and prev_active != active_meme_id:
            fading_out_id = prev_active
            # Fade out old audio when display target changes
            audio_slot.stop(prev_active)

        if active_meme_id is not None:
            fade_alpha = min(255, fade_alpha + fade_speed)
            if fade_alpha >= 255:
                fading_out_id = None  # New image fully visible, discard old
        elif fading_out_id is not None:
            fade_alpha = max(0, fade_alpha - fade_speed)
            if fade_alpha == 0:
                fading_out_id = None  # Fade-out complete
                # Audio fade-out already initiated by stop() when hold ended
        else:
            fade_alpha = max(0, fade_alpha - fade_speed)

        # ?? Audio volume sync (delegated to AudioSlot state machine) ??
        audio_slot.tick(dt * 1000.0)

        # --- ?? ---
        # ????????
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bg_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
        screen.blit(pygame.transform.scale(bg_surface, (screen_w, screen_h)), (0, 0))

        # Debug??????
        if debug_cfg.get("draw_landmarks", False) and results is not None:
            _draw_landmarks(screen, results, screen_w, screen_h, frame.shape)

        # Debug?????
        if debug_cfg.get("show_status_text", False):
            font = pygame.font.SysFont("consolas", 18)
            status_lines = [
                f"State: {state_machine.state.name}",
                f"Debounce: {state_machine.debounce_progress:.0%}",
                f"Features: {', '.join(f'{k}={v}' for k,v in fv.features.items())}",
                f"Frame: {frame_id}",
                f"Audio: {len(all_keywords)} kw, {'ON' if vosk_engine.is_available else 'OFF'}",
                f"Display: {'hold:'+hold_display_id if hold_display_id else 'none'}, 1shot:{len(one_shot_items)}",
                f"Owner: {hold_owner_feature or 'none'}",
            ]
            y = 10
            for line in status_lines:
                surf = font.render(line, True, (0, 255, 0))
                screen.blit(surf, (10, y))
                y += 22

        # ????
        display_id = active_meme_id or fading_out_id
        if display_id and fade_alpha > 0:
            meme_img = default_images.get(display_id)
            if meme_img is None:
                # Look up image path from mapping or one_shot queue
                img_path = None
                m = _mapping_lookup.get(display_id)
                if m:
                    img_path = m.image_path
                if img_path is None:
                    for item in one_shot_items:
                        if item.get("mapping_id") == display_id:
                            img_path = item.get("image")
                            break
                if img_path:
                    meme_img = load_image(img_path)
                    if meme_img is not None:
                        default_images[display_id] = meme_img
                        if debug_cfg.get("trace_renderer", False):
                            log_trace("render", "renderer.pygame", "asset_loaded",
                                      path=img_path, size=f"{meme_img.get_width()}x{meme_img.get_height()}")
                    elif debug_cfg.get("trace_renderer", False):
                        log_trace("render", "renderer.pygame", "render_failed",
                                  reason="asset_load_failed", path=img_path or "null")
            if meme_img is not None:
                meme_surface = meme_img.copy()
                meme_surface.set_alpha(fade_alpha)
                screen.blit(
                    meme_surface,
                    meme_surface.get_rect(center=(screen_w // 2, screen_h // 2)),
                )

        pygame.display.flip()
        frame_id += 1

    # --- ?? ---
    audio_slot.shutdown()
    camera.release()
    holistic.close()
    vosk_engine.stop()
    pygame.quit()
    log.info("AutoMeme ??")


def _draw_landmarks(screen, results, sw: int, sh: int, frame_shape):
    """Debug??? MediaPipe ????"""
    import pygame

    h, w = frame_shape[:2]
    scale_x = sw / w
    scale_y = sh / h
    color = (0, 255, 0)

    def pt(lm):
        return (int(lm.x * w * scale_x), int(lm.y * h * scale_y))

    # Pose
    if results.pose_landmarks and results.pose_landmarks.landmark:
        lm_list = results.pose_landmarks.landmark
        for connection in [(11, 12), (11, 23), (12, 24), (23, 24),
                           (23, 25), (24, 26), (25, 27), (26, 28)]:
            if connection[0] < len(lm_list) and connection[1] < len(lm_list):
                a = pt(lm_list[connection[0]])
                b = pt(lm_list[connection[1]])
                pygame.draw.line(screen, color, a, b, 2)

    # Hands
    for hand_lm in [results.left_hand_landmarks, results.right_hand_landmarks]:
        if hand_lm is None:
            continue
        lm_list = hand_lm.landmark
        for connection in [(0, 1), (1, 2), (2, 3), (3, 4),           # thumb
                           (0, 5), (5, 6), (6, 7), (7, 8),           # index
                           (0, 9), (9, 10), (10, 11), (11, 12),      # middle
                           (0, 13), (13, 14), (14, 15), (15, 16),    # ring
                           (0, 17), (17, 18), (18, 19), (19, 20),    # pinky
                           (5, 9), (9, 13), (13, 17)]:
            if connection[0] < len(lm_list) and connection[1] < len(lm_list):
                a = pt(lm_list[connection[0]])
                b = pt(lm_list[connection[1]])
                pygame.draw.line(screen, (255, 255, 0), a, b, 1)

    # Face mesh (simplified)
    if results.face_landmarks and results.face_landmarks.landmark:
        for lm in results.face_landmarks.landmark:
            x, y = pt(lm)
            if 0 <= x < sw and 0 <= y < sh:
                pygame.draw.circle(screen, (0, 128, 255), (x, y), 1)


if __name__ == "__main__":
    main()
