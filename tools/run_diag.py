"""一键诊断入口。
用法: python tools/run_diag.py [voice|mapping|trigger <feature_id>|validate]
"""
import sys, os, subprocess

TOOLS = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(TOOLS)

cmds = {
    "voice": f'python "{os.path.join(TOOLS, "voice_debug.py")}"',
    "validate": f'python "{os.path.join(TOOLS, "validate_config.py")}"',
}

if len(sys.argv) < 2:
    print("Usage: python tools/run_diag.py <command>")
    print("Commands:")
    print("  voice       - Test microphone + Vosk + keyword detection")
    print("  validate    - Validate config files")
    print("  trigger <id> - Simulate trigger for a feature (e.g. is_donk)")
    sys.exit(1)

cmd = sys.argv[1]
if cmd == "trigger":
    if len(sys.argv) < 3:
        print("Usage: python tools/run_diag.py trigger <feature_id>")
        sys.exit(1)
    subprocess.run(["python", os.path.join(TOOLS, "trigger_debug.py"), sys.argv[2]], cwd=PROJECT)
elif cmd in cmds:
    subprocess.run(cmds[cmd], shell=True, cwd=PROJECT)
else:
    print(f"Unknown command: {cmd}")
    sys.exit(1)
