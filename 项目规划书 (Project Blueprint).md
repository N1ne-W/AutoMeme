# 项目规划书 (Project Blueprint)

**项目代号：**基于ai图像识别的自动吊图跳脸器

## 1. 初衷与动机 (The "Why")

通过动作识别触发吊图感觉很有意思

## 2. 核心功能 (MVP - Minimum Viable Product)

- **必须有 (Must Have)：**
  1. 图像识别自动触发吊图
  2. 语音识别自动触发吊图/bgm
- **可以有 (Nice to Have - 以后再说)：**
  1. 自定义 ==动作-吊图== ==语音-bgm== 关系对应

## 3. 技术栈游乐场 (Tech Stack Playground)

*   **核心语言：** Python 3.x
*   **视觉识别 (Vision)：** **OpenCV** (读取摄像头) + **MediaPipe** (谷歌开源的神器，毫秒级识别骨骼、手势，不需要显卡也能跑得飞快)。
*   **语音识别 (Audio)：** **SpeechRecognition** 库 (配合 Vosk 或 PocketSphinx 做离线唤醒，延迟低) 或者简单的 **PyAudio** 监测音量波形。
*   **GUI/展示 (Display)：** **PyGame** (推荐，适合做全屏图片/音频展示，处理多媒体非常流畅) 或 **Tkinter** (如果只是简单弹窗)。

## 4. 极简架构/逻辑 (Architecture Lite)

### 4.1 系统逻辑流程 (The Flow)
整个程序是一个**并行**的结构，主要分为“主循环（视觉+UI）”和“后台线程（听觉）”。

```text
[主线程：眼睛与画布]                  [子线程：耳朵]
       |                                   |
1. 初始化 PyGame 窗口                  1. 开启麦克风监听
2. 打开摄像头 (OpenCV)                 2. 持续识别关键词 (While True)
       |                                   |
       v                                   v
[ 进入死循环 Loop ] <---------------- [ 监听到关键词? ]
       |                                   |
   A. 读取一帧画面                         |
   B. 扔给 MediaPipe 识别动作/手势          |
   C. 判断逻辑 (Matching Logic)            |
       |--> 检测到 "剪刀手"? ---> 触发事件A |
       |--> 检测到 "挥手"?   ---> 触发事件B |
       |<-- 收到音频信号?    <--- 触发事件C |
       |                                   |
   D. 更新 PyGame 画面 (渲染图片)           |
   E. 播放音频 (如有)                       |
       |                                   |
[ 回到 A 继续下一帧 ]                      |
```

### 4.2 关键数据结构 (配置映射表)
不要把逻辑写死在代码里（Hardcode），建议用一个字典（Dictionary）来管理动作和资源的对应关系，这样你以后想换图很方便。

```python
# config.py - 你的配置文件

EVENT_MAPPING = {
    # 视觉触发配置
    "GESTURE_VICTORY": {  # 识别到“剪刀手”
        "type": "image",
        "path": "./assets/cool_guy.png",
        "duration": 3.0   # 显示3秒
    },
    "POSE_SQUAT": {       # 识别到“下蹲”
        "type": "audio",
        "path": "./assets/fart_sound.mp3"
    },
  
    # 语音触发配置
    "KEYWORD_OPEN": {     # 喊出“芝麻开门”
        "type": "image",
        "path": "./assets/treasure.jpg"
    }
}
```

### 4.3 核心难点与伪代码 (Core Logic)

这里有一个资深经验分享：**去抖动 (Debouncing)**。
机器识别很快，可能会在0.1秒内识别出“有手势”->“没手势”->“有手势”，导致图片疯狂闪烁。你需要一个计数器。

```python
# 伪代码逻辑示意

current_gesture = None
gesture_counter = 0
THRESHOLD = 5 # 连续5帧都识别到同一个动作，才算确认

while True:
    # 1. 视觉处理
    frame = camera.read()
    raw_gesture = model.predict(frame) # MediaPipe 返回结果
  
    # 2. 防抖逻辑 (Stable Check)
    if raw_gesture == last_raw_gesture:
        gesture_counter += 1
    else:
        gesture_counter = 0
      
    if gesture_counter > THRESHOLD:
        final_gesture = raw_gesture
      
    # 3. 触发与渲染
    if final_gesture == "Victory_Hand":
        screen.show(EVENT_MAPPING["GESTURE_VICTORY"]["path"])
    elif audio_queue.has_msg("芝麻开门"):
        screen.show(EVENT_MAPPING["KEYWORD_OPEN"]["path"])
      
    # 4. 刷新屏幕
    pygame.display.update()
```

## 5. 开发待办清单 (To-Do List)

###  Phase 0: 环境准备与地基 (Infrastructure)

> **目标：** 搭建好开发环境，确保“也就是跑个 Hello World”的程度。

-  **创建项目文件夹**：建立清晰的目录结构（如 `/src`, `/assets`, `/tests`）。
-  **初始化 Git 仓库**：`git init`，创建 `.gitignore` (排除 `__pycache__`, `venv`)，并添加 **MIT License** 文件。
-  **配置虚拟环境**：
  -  创建 Python 虚拟环境 (`python -m venv venv`)。
  -  激活虚拟环境。
-  **安装核心依赖**：创建一个 `requirements.txt`。
  -  `pip install opencv-python mediapipe pygame SpeechRecognition pyaudio`
-  **准备素材**：找两张测试图片（例如 `ok.png`, `stop.png`）和一个测试音频，放入 `/assets` 文件夹。

------

### 👁️ Phase 1: 视觉核心 MVP (Vision Prototype)

> **目标：** 让电脑“看见”你的手，并在控制台打印结果（先别管界面显示）。

-  **摄像头连通性测试**：写一个脚本，使用 OpenCV 仅仅打开摄像头并显示画面。
-  **集成 MediaPipe Hands**：
  -  在摄像头画面上绘制出手部骨骼关键点。
-  **编写手势判断逻辑 (Gesture Logic)**：
  -  **数学计算**：写一个函数，通过计算手指坐标距离，判断“手指是伸直还是弯曲”。
  -  **简单识别**：识别简单的“张开手掌 (Open Hand)”和“拳头 (Fist)”。
-  **控制台输出验证**：当检测到拳头时，在终端 print("检测到拳头")。

------

### 🖼️ Phase 2: 图形界面对接 (GUI Integration)

> **目标：** 告别黑底白字的控制台，让图片根据手势在窗口中显示。

-  **PyGame 基础窗口**：初始化 PyGame，弹出一个空白窗口。
-  **视频流嵌入**：将 OpenCV 读取到的画面转换格式，实时渲染到 PyGame 窗口背景中（这一步有点坑，注意颜色格式 BGR 转 RGB）。
-  **实现触发器 (Trigger)**：
  -  如果 Phase 1 的逻辑返回“拳头”，在 PyGame 窗口上层覆盖显示 `stop.png`。
  -  如果没有手势，隐藏图片。
-  **添加去抖动 (Debouncing)**：实现一个简单的计数器，防止手势识别不稳定导致的图片闪烁（Flash）。

------

### 👂 Phase 3: 听觉子系统 (Audio Subsystem)

> **目标：** 让电脑“听见”关键词。这一步独立开发，不要先混入主程序。

-  **麦克风测试**：写一个独立脚本 `test_mic.py`，确认能录音。
-  **语音转文字 (STT)**：集成 `SpeechRecognition` 库（推荐先用 Google Web API 测试，或者 PocketSphinx 离线版）。
-  **关键词匹配**：
  -  编写逻辑：如果识别到的文本包含 "芝麻开门"，print("触发！")。
-  **多线程实现 (Threading)**：
  -  **重点难点**：语音识别通常是阻塞的（Listening 时程序会卡住）。
  -  封装一个 `AudioListener` 类，继承自 `threading.Thread`，让它在后台默默监听，把结果写入一个公共队列 (Queue)。

------

### 🧠 Phase 4: 完整集成 (System Integration)

> **目标：** 将 视觉、UI、听觉 融合在一起。

-  **配置文件化**：创建 `config.py`，定义好 `动作 -> 图片路径` 和 `语音 -> 图片路径` 的字典。
-  **合并线程**：在 PyGame 的主循环中，加入对“听觉线程”消息队列的检查。
  -  逻辑：`if not audio_queue.empty(): 处理语音指令`
-  **优先级仲裁**：决定如果同时检测到手势和语音，谁说了算？（通常语音作为一次性触发，手势作为持续触发）。
-  **资源预加载**：程序启动时把图片都加载到内存，避免每次触发时读取硬盘导致的卡顿。

------

### ✨ Phase 5: 优化与交付 (Polish)

> **目标：** 让软件好用、稳定。

-  **UI 美化**：给识别到的手画个酷炫的框，或者加点半透明特效。
-  **异常处理**：
  -  拔掉摄像头程序不崩溃。
  -  没麦克风也能跑视觉功能。
-  **代码重构**：删除调试用的 print，添加注释。
-  **编写 README**：把刚才写的项目简介和运行方法写进去。

## 6. 资源与参考 (Resources)

- 参考项目链接 (GitHub)
- API 文档地址
- 设计灵感来源 (Dribbble/Pinterest)
- github仓库地址：https://github.com/N1ne-W/AI-based-image-speech-recognition-automatic-meme-BGM-face-changer