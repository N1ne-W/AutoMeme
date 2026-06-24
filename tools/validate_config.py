"""配置校验器：检查 features.json + mappings.json 完整性。
用法: python tools/validate_config.py
"""
import sys, os, json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
errors = []
warnings = []

def check(cond, msg, level="error"):
    (errors if level == "error" else warnings).append(f"[{level.upper()}] {msg}")

# Load features
feat_path = os.path.join(PROJECT_ROOT, "config", "features.json")
if not os.path.exists(feat_path):
    check(False, f"features.json not found: {feat_path}")
    features = {}
else:
    with open(feat_path, encoding="utf-8") as f:
        feat_cfg = json.load(f)
    features = feat_cfg.get("features", {})
    print(f"Features: {len(features)} loaded")
    for fid, fdef in features.items():
        mod = fdef.get("module", "")
        if not mod:
            check(False, f"Feature '{fid}' has no module")
        feat_file = os.path.join(PROJECT_ROOT, "src", "vision", "features", f"{mod}.py")
        if not os.path.exists(feat_file):
            check(False, f"Feature '{fid}': module file not found: {feat_file}")

# Load mappings
map_path = os.path.join(PROJECT_ROOT, "config", "mappings.default.json")
if not os.path.exists(map_path):
    check(False, f"mappings.default.json not found: {map_path}")
    mappings = []
else:
    with open(map_path, encoding="utf-8") as f:
        map_cfg = json.load(f)
    mappings = map_cfg.get("mappings", [])
    print(f"Mappings: {len(mappings)} loaded")

# Validate mappings
seen_ids = set()
for m in mappings:
    mid = m.get("id", "?")
    if mid in seen_ids:
        check(False, f"Duplicate mapping id: {mid}")
    seen_ids.add(mid)

    # Feature references
    for fid in m.get("conditions", {}).get("features", []):
        if fid not in features:
            check(False, f"Mapping '{mid}': feature '{fid}' not in features.json")

    # Voice keywords
    for kw in m.get("conditions", {}).get("voice_keywords", []):
        if not kw.strip():
            warnings.append(f"[WARN] Mapping '{mid}': empty keyword")

    # Asset path
    img = m.get("actions", {}).get("image")
    if img:
        img_full = os.path.join(PROJECT_ROOT, "assets", img.replace("/", os.sep))
        if not os.path.exists(img_full):
            check(False, f"Mapping '{mid}': asset not found: {img_full}")

    # Valid priority
    pri = m.get("priority", 0)
    if not isinstance(pri, (int, float)) or pri < 0:
        check(False, f"Mapping '{mid}': invalid priority: {pri}")

    # Valid cooldown
    cd = m.get("cooldown_ms", 0)
    if not isinstance(cd, (int, float)) or cd < 0:
        check(False, f"Mapping '{mid}': invalid cooldown_ms: {cd}")

    # Valid display_mode
    dm = m.get("display_mode", "hold")
    if dm not in ("hold", "duration"):
        check(False, f"Mapping '{mid}': invalid display_mode: {dm} (expected 'hold' or 'duration')")
    if dm == "duration":
        dur = m.get("duration_ms", 0)
        if not isinstance(dur, (int, float)) or dur <= 0:
            check(False, f"Mapping '{mid}': duration mode requires duration_ms > 0")

# Report
print()
if errors:
    print(f"ERRORS ({len(errors)}):")
    for e in errors:
        print(f"  {e}")
if warnings:
    print(f"WARNINGS ({len(warnings)}):")
    for w in warnings:
        print(f"  {w}")
if not errors and not warnings:
    print("All checks passed!")
print()
print(f"Result: {len(errors)} errors, {len(warnings)} warnings")
sys.exit(1 if errors else 0)
