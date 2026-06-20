# AutoMeme — 基于 AI 视觉融合识别的自动吊图跳脸器

AutoMeme 是一个面向直播/视频互动的**多模态动作识别工具**。通过摄像头捕捉手势与肢体姿态、麦克风监听语音关键词，实时在画面叠加层上弹出对应的表情包（PNG/GIF）和音效。

## 核心能力

- **视觉识别**：OpenCV 采集 + MediaPipe Holistic 提取全身关键点（手部 21 点 + 身体 33 点 + 面部 468 点）
- **语音识别**：Vosk 离线引擎，支持中文关键词唤醒
- **特征向量判定引擎**：布尔特征 + 状态机，支持组合动作判定 + 去抖动
- **无边框透明叠加层**：可叠加在 OBS / 游戏画面上，不影响原有画面
- **组合编辑器（GUI）**：拖动配置「动作 + 语音 = 吊图 + 音效」映射

## 快速启动

```bash
git clone https://github.com/N1ne-W/AutoMeme.git
cd AutoMeme
python -m venv venv
.\venv\Scripts\activate   # Windows
pip install -r requirements.txt
python main.py
```

## 文档索引

| 文档 | 内容 |
|------|------|
| [CONTEXT.md](CONTEXT.md) | 领域术语表 |
| [docs/01-project-overview.md](docs/01-project-overview.md) | 项目说明、范围、里程碑、风险 |
| [docs/02-requirements-and-acceptance.md](docs/02-requirements-and-acceptance.md) | 功能需求、非功能需求、验收标准 |
| [docs/03-architecture-and-technical-design.md](docs/03-architecture-and-technical-design.md) | 架构设计、技术选型、ADR |
| [docs/04-api-and-data-design.md](docs/04-api-and-data-design.md) | 模块接口、配置格式、数据设计 |
| [docs/05-deployment-operations-and-monitoring.md](docs/05-deployment-operations-and-monitoring.md) | 部署、打包、日志、监控 |
| [docs/06-testing-release-and-changelog.md](docs/06-testing-release-and-changelog.md) | 测试计划、用例、发布说明 |

## 技术栈

| 组件 | 技术 |
|------|------|
| 视觉识别 | OpenCV + MediaPipe Holistic |
| 语音识别 | Vosk（离线），备选 SpeechRecognition |
| GUI / 渲染 | PyGame（无边框透明叠加窗口） |
| 打包分发 | PyInstaller / Nuitka -> Windows 安装程序 |
| 语言 | Python 3.10+ |

## 许可证

GNU General Public License v3.0 (GPLv3)
