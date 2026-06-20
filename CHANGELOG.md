# CHANGELOG

## [v0.1.0] - 2026-06-20

### Added
- 模块化视觉管道 `src/vision/`：Camera / HolisticRunner / FeatureExtractor
- 5 个手势特征检测器：Donk、MonkeyThink、NFB、OMG、ThumbsUp
- 3 个通用特征检测器：Squat、Victory、Heart
- 判定引擎 `src/engine/`：状态机 / 去抖动 / 冷却 / 映射引擎
- 配置体系 `config/`：app.yaml / features.json / mappings.default.json / mappings.user.json
- 全链路集成 main.py（PyGame 渲染 + MediaPipe Debug 骨骼线）
- 特征向量 + 状态机架构，支持 JSON 配置驱动的映射表
- 素材双目录管理：default/（内置） + user/（用户自定义）
- 36 个单元测试覆盖引擎和特征检测器
- 兼容 MediaPipe 0.10+ API（LegacyResults 包装层）

### Changed
- 判定引擎从直接坐标判定重构为特征向量 + 状态机架构
- 硬编码路径改为动态项目根路径
- 素材目录从 assets/images/ 迁移到 assets/default/images/

### Fixed
- 去抖动 progress 封顶 1.0
- ThumbsUp 误触：增加拇指远离食指检查，区分握拳和点赞
- Debug 骨骼线补全拇指连接（0→1→2→3→4）

## [v0.0.0] - Unreleased (legacy)
- 原始原型：单文件 vision.py 坐标判定