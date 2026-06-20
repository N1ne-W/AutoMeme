# AutoMeme 🎭

> AI 视觉融合识别的自动表情包触发工具 — 面向直播与视频互动

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-GPLv3-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Architecture%20Refactor-yellow)](https://github.com/N1ne-W/AutoMeme)

AutoMeme 通过摄像头实时捕捉你的手势和肢体动作，自动在画面上弹出对应的表情包。无需手动操作，让直播互动更有趣。

---

## ✨ 功能

- 🙌 **5 种手势识别** — OMG / NFB / Donk / MonkeyThink / 👍 点赞
- 🧠 **特征向量判定引擎** — 状态机 + 去抖动 + 冷却，精准不误触
- ⚙️ **JSON 配置驱动** — 映射表可自定义，新增手势不修改引擎代码
- 🎨 **PyGame 实时渲染** — 表情包淡入淡出，MediaPipe 骨骼线 Debug
- 🔊 **语音触发预留** — Vosk 离线识别接口已就绪（Phase 3）

## 🎮 当前支持的手势

| 手势              | 触发方式                     |
| ----------------- | ---------------------------- |
| 🎉 **OMG**         | 双手张开 + 靠近耳朵 + 张嘴   |
| 🖐️ **NFB**         | 单手张开 + 靠近耳朵 + 闭嘴   |
| 👃 **Donk**        | 食指尖靠近鼻子               |
| 🐵 **MonkeyThink** | 食指尖靠近嘴角（不靠近鼻子） |
| 👍 **ThumbsUp**    | 握拳 + 拇指竖起              |

## 🚀 快速开始

```bash
git clone https://github.com/N1ne-W/AutoMeme.git
cd AutoMeme

# 创建虚拟环境（推荐 Python 3.10+）
python -m venv venv
.\venv\Scripts\activate    # Windows
source venv/bin/activate   # macOS / Linux

# 安装依赖
pip install -r requirements.txt

# 下载 MediaPipe Holistic 模型
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task" -OutFile "assets/models/holistic_landmarker.task"

# 运行
python main.py
```

> **操作提示**：`ESC` 退出 | 左上角显示实时状态和特征值 | 绿色骨骼线 = 身体 | 黄色手掌线 = 手部

## 🏗️ 项目结构

```
src/
├── vision/              # 视觉管道
│   ├── camera.py        #   摄像头采集
│   ├── holistic_runner.py # MediaPipe Holistic 推理
│   ├── feature_extractor.py # 特征向量提取
│   └── features/        #   手势检测器（5 + 3 通用）
├── engine/              # 判定引擎
│   ├── state_machine.py #   IDLE → DETECTING → TRIGGERED → COOLDOWN
│   ├── debounce.py      #   去抖动（连续N帧确认）
│   ├── cooldown.py      #   冷却计时
│   ├── mapping_engine.py #  JSON 映射表匹配
│   └── signals.py       #   信号/事件定义
├── audio/               # 语音管道（Phase 3）
├── renderer/            # 渲染层（Phase 4）
└── editor/              # 组合编辑器（Phase 5）

config/
├── app.yaml             # 应用参数
├── features.json        # 特征注册表
├── mappings.default.json # 默认映射
└── mappings.user.json   # 用户自定义映射

assets/
├── default/images/      # 内置表情包
├── default/audio/       # 内置音效
├── user/images/         # 用户素材
├── user/audio/
└── models/              # MediaPipe / Vosk 模型
```

## 🧪 测试

```bash
pip install pytest
python -m pytest tests/ -v
```

## 📋 技术栈

| 组件     | 技术                                |
| -------- | ----------------------------------- |
| 视觉识别 | OpenCV + MediaPipe Holistic (0.10+) |
| 语音识别 | Vosk（离线，计划中）                |
| GUI 渲染 | PyGame                              |
| 打包分发 | PyInstaller / NSIS                  |
| 配置格式 | JSON + YAML                         |
| 测试     | pytest                              |

## 🗺️ Roadmap

- [x] Phase 0 — 文档体系 + 项目初始化
- [x] Phase 1 — 模块化视觉管道 + 特征检测器
- [x] Phase 2 — 判定引擎（状态机 + 映射）
- [ ] Phase 3 — Vosk 语音关键词触发
- [ ] Phase 4 — 无边框透明叠加层 + GIF 支持
- [ ] Phase 5 — GUI 组合编辑器
- [ ] Phase 6 — Windows 安装程序

## 📄 License

GNU General Public License v3.0
