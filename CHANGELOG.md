# CHANGELOG


## [v0.2.0] - 2026-06-24

### Added
- GestureStabilizer: time-based enter/exit debounce for hold display (src/engine/gesture_stabilizer.py)
- FeatureState machine: INACTIVE -> ENTERING -> ACTIVE -> LOST_GRACE -> INACTIVE
- 20 gesture stabilizer tests (enter/exit debounce, lost grace recovery, owner safety)
- AudioSlot: audio state machine with idempotent play, fade-in/out (src/engine/audio_slot.py)
- AudioState machine: IDLE -> FADING_IN -> PLAYING -> FADING_OUT -> IDLE
- 16 audio slot tests (idempotent play, owner safety, config defaults)
- Image fade-out effect: fading_out_id maintains display during alpha decay
- Config: gesture section (enter/exit_debounce_ms, min_hold_ms) in app.yaml
- Config: audio section (follow_image_lifecycle, loop_when_image_alive, fade_in/out_ms) in app.yaml
- Mapping: display_mode, duration_ms, enter/exit_debounce_ms, min_hold_ms fields
- Donk mapping: audio=sounds/donk.mp3, mode=both

### Changed
- Debounce: hard reset (counter=0) -> leaky bucket decay (counter = max(0, counter-1))
- Hold display: frame-based counter -> time-based GestureStabilizer with owner tracking
- Audio: inline _start_audio/_stop_audio -> AudioSlot state machine
- Renderer: active_meme_id fallback -> display_id = active_meme_id or fading_out_id
- Test count: 49 -> 85 (36 new tests)

### Fixed
- Feature oscillation causing hold display flicker: time-based exit debounce absorbs jitter
- Audio overlapping on repeated gesture triggers: AudioSlot.play() is idempotent
- Image disappearing without fade-out: fading_out_id preserves old image during alpha decay
- Audio not stopping when image disappears: AudioSlot.stop() with owner check
- Old config without gesture/audio sections: defaults applied, no crash


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