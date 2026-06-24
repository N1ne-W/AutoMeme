"""语音管道调试工具：测试麦克风 + Vosk + 关键词。
用法: python tools/voice_debug.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from src.diagnostics import init_debug_config, new_trace_id, log_trace
from src.audio import VoskEngine, extract_keywords

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load config
with open(os.path.join(PROJECT_ROOT, "config", "app.yaml"), encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
init_debug_config(cfg.get("debug", {}))

vosk_cfg = cfg.get("vosk", {})
model_full = os.path.join(PROJECT_ROOT, vosk_cfg.get("model_path", ""))

keywords = extract_keywords(os.path.join(PROJECT_ROOT, "config", "mappings.default.json"))
user_kw = extract_keywords(os.path.join(PROJECT_ROOT, "config", "mappings.user.json"))
all_keywords = list(set(keywords + user_kw))

print("=" * 50)
print("Voice Debug Tool")
print("=" * 50)
print(f"Model path : {model_full}")
print(f"Model exists: {os.path.exists(model_full)}")
print(f"Keywords   : {all_keywords}")
print()

print("Starting Vosk engine...")
engine = VoskEngine(model_path=model_full, keywords=all_keywords)
ok = engine.start()

if not ok:
    print(f"FAILED: last_error={engine.last_error}")
    print("Check: model path, keywords in mappings, microphone access")
    sys.exit(1)

print("Listening... Say a keyword and press Ctrl+C to stop.")
print()
try:
    while True:
        sig = engine.poll()
        if sig:
            print(f"[{sig.trace_id}] KEYWORD: '{sig.keyword}' (conf={sig.confidence:.2f}) text='{sig.text}'")
        time.sleep(0.1)
except KeyboardInterrupt:
    print()
    print("Stopped.")
finally:
    engine.stop()
