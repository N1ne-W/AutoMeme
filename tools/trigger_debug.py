"""触发链路调试：模拟 feature 事件走完整链路。
用法: python tools/trigger_debug.py <feature_id>
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from src.diagnostics import init_debug_config, new_trace_id, log_trace
from src.engine.mapping_engine import MappingEngine
from src.engine.state_machine import StateMachine, EngineState
from src.engine.signals import VisionSignal

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Config
with open(os.path.join(PROJECT_ROOT, "config", "app.yaml"), encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
init_debug_config(cfg.get("debug", {}))

if len(sys.argv) < 2:
    print("Usage: python tools/trigger_debug.py <feature_id>")
    print("Available features from features.json:")
    with open(os.path.join(PROJECT_ROOT, "config", "features.json"), encoding="utf-8") as f:
        feats = json.load(f)["features"]
    for fid in feats:
        print(f"  {fid}")
    sys.exit(1)

feature_id = sys.argv[1]
trace_id = new_trace_id()

print("=" * 50)
print(f"Trigger Debug: feature_id={feature_id}")
print(f"Trace ID: {trace_id}")
print("=" * 50)

# Load features config
with open(os.path.join(PROJECT_ROOT, "config", "features.json"), encoding="utf-8") as f:
    features_cfg = json.load(f)["features"]

if feature_id not in features_cfg:
    print(f"ERROR: feature '{feature_id}' not in features.json")
    print(f"Available: {list(features_cfg.keys())}")
    sys.exit(1)

print(f"Feature found: {features_cfg[feature_id]['description']}")

# Mapping engine
me = MappingEngine()
me.load(
    os.path.join(PROJECT_ROOT, "config", "mappings.default.json"),
    os.path.join(PROJECT_ROOT, "config", "mappings.user.json"),
)
print(f"Loaded {len(me._mappings)} mappings")

# Check match
result = me.match_vision({feature_id: True})
if result is None:
    print(f"NO MAPPING for feature '{feature_id}'")
    print("Available mappings and their features:")
    for m in me._mappings:
        if m.enabled:
            print(f"  {m.id} -> {m.features}")
    sys.exit(1)

print(f"Matched mapping: {result.id}")
print(f"  Asset: {result.image_path}")
asset_full = os.path.join(PROJECT_ROOT, "assets", result.image_path.replace("/", os.sep))
print(f"  Asset exists: {os.path.exists(asset_full)}")
print(f"  Priority: {result.priority}")
print(f"  Cooldown: {result.cooldown_ms}ms")
print(f"  Debounce: {result.debounce_frames} frames")

# State machine
sm = StateMachine(me)
print()
print("Simulating state machine (8 frames)...")
result_event = None
for i in range(result.debounce_frames):
    result_event = sm.update(
        vision_signal=VisionSignal(
            features={feature_id: True},
            frame_id=i, timestamp=time.time(),
        ),
        trace_id=trace_id,
    )
    state = sm.state.name
    if result_event:
        print(f"  Frame {i}: {state} -> TRIGGERED! mapping={result_event.mapping_id}")
    else:
        print(f"  Frame {i}: {state}")

if result_event:
    print()
    print("SUCCESS: Trigger would fire!")
    print(f"  mapping_id: {result_event.mapping_id}")
    print(f"  action_type: {result_event.action_type}")
    print(f"  image: {result_event.image_path}")
else:
    print()
    print("FAILED: No trigger event produced")
