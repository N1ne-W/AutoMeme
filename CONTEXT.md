# CONTEXT.md — AutoMeme 领域术语表

> 本文档定义项目中的核心领域术语（Ubiquitous Language）。不包含实现细节。

## 核心概念

| 术语 | 英文 | 定义 |
|------|------|------|
| 吊图 | Meme / Reaction Image | 由动作或语音触发的图片（PNG）或动图（GIF），叠加在直播画面上 |
| 触发 | Trigger | 用户通过特定动作组合或语音关键词激活吊图弹出的事件 |
| 动作词汇 | Action Vocabulary | 可被系统识别的预定义手势/肢体姿态组合 |
| 特征向量 | Feature Vector | 将复杂动作拆解为一组布尔标志位，供判定引擎消费 |
| 判定引擎 | Judgment Engine | 消费特征向量，根据映射表决定当前应触发哪个吊图的状态机 |
| 映射表 | Mapping Table | 定义动作/语音到吊图+音效的对应关系，支持用户自定义 |
| 去抖动 | Debounce | 要求连续 N 帧识别到同一动作才确认触发 |
| 冷却时间 | Cooldown | 同一吊图两次触发之间的最小间隔 |
| 叠加层 | Overlay | 无边框透明窗口，置顶渲染吊图 |
| 跳脸 | Jump-Scare / Pop-up | 吊图在直播画面上突然弹出的视觉表现 |

## 素材

| 术语 | 英文 | 定义 |
|------|------|------|
| 默认素材包 | Default Asset Pack | 随安装程序分发的内置吊图和音效集合 |
| 用户素材 | User Assets | 用户自行添加到素材目录的自定义文件 |
| BGM | Sound Effect | 与吊图同时触发的音效（MP3/WAV） |

## 系统角色

| 术语 | 英文 | 定义 |
|------|------|------|
| 摄像头输入 | Camera Input | 实时视频流，来源为系统默认摄像头 |
| 麦克风输入 | Microphone Input | 实时音频流 |
| 视觉管道 | Vision Pipeline | OpenCV + MediaPipe Holistic 关键点提取管道 |
| 听觉管道 | Audio Pipeline | Vosk 离线语音识别引擎 |
| 渲染器 | Renderer | PyGame 驱动的无边框透明叠加窗口 |
| 组合编辑器 | Combo Editor | GUI 面板，允许用户自定义映射关系 |

## 状态

| 状态 | 说明 |
|------|------|
| IDLE | 无动作触发，画面仅显示摄像头 |
| DETECTING | 检测到候选动作，等待去抖动确认 |
| TRIGGERED | 动作确认，吊图渲染中 |
| COOLDOWN | 吊图淡出后进入冷却期 |
