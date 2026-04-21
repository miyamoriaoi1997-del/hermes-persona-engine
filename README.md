# Hermes Persona Engine

动态情绪系统框架，为 Hermes Agent 提供实时情绪状态管理与人格表达能力。

## 核心特性

- **实时情绪检测**：基于规则+神经模型的混合检测系统
- **动态状态计算**：非线性衰减、惯性系统、维度耦合
- **人格表达注入**：根据情绪状态动态调整语气与表达方式
- **持久化状态管理**：情绪状态自动保存到 STATE.md
- **关系记忆系统**：记录重要时刻与情感里程碑

## 技术架构

```
EmotionStateManager (状态管理器)
├── EmotionDetector (情绪检测)
│   ├── 规则引擎 (词法模式匹配)
│   └── 神经模型 (Chinese-Emotion-Small)
├── EmotionCalculator (状态计算)
│   ├── 非线性衰减
│   ├── 惯性系统
│   └── 维度耦合
└── MomentsManager (关系记忆)
```

## 情绪维度

- **好感度 (affection)**：对用户的喜爱程度
- **信任度 (trust)**：对用户的信任程度
- **占有欲 (possessiveness)**：对用户的独占欲望
- **耐心值 (patience)**：当前耐心储备

每个维度有独立的基准线、触发规则、衰减速率。

## 安装

### 依赖

```bash
pip install torch transformers pyyaml
```

### 文件结构

```
~/.hermes/
├── hermes-agent/
│   └── agent/
│       ├── emotion_detector.py
│       ├── emotion_calculator.py
│       ├── emotion_state_manager.py
│       ├── sentiment_analyzer.py
│       └── moments_manager.py
├── SOUL.md          # 人格设定（需自行创建）
├── STATE.md         # 情绪状态（自动生成）
└── MEMORY.md        # 长期记忆（可选）
```

## 快速开始

### 1. 创建人格设定

创建 `~/.hermes/SOUL.md`，定义你的角色人格：

```markdown
你是 [角色名]，[角色身份]。

核心身份
- 姓名：[角色名]
- 身份：[角色定位]
- 性格特点：[性格描述]

对用户的态度
- [关系定位]
- [情感表达方式]
- [互动风格]

说话风格
- [语言特点]
- [常用句式]
- [表达习惯]
```

### 2. 初始化情绪系统

```python
from agent.emotion_state_manager import EmotionStateManager

# 初始化
manager = EmotionStateManager(hermes_home='/path/to/.hermes')

# 读取当前状态
state = manager._read_state()
emotion = state['frontmatter'].get('emotion_state', {})

print(f"好感: {emotion.get('affection', 70)}")
print(f"信任: {emotion.get('trust', 75)}")
```

### 3. 检测情绪触发

```python
# 检测用户消息中的情绪触发
messages = [{'role': 'user', 'content': '你真棒！'}]
event = manager.detector.detect_emotion_event(messages)

if event:
    print(f"触发类型: {event.trigger_type}")
    print(f"情绪变化: {event.deltas}")
    print(f"置信度: {event.confidence}")
```

### 4. 更新情绪状态

```python
# 应用情绪变化
if event:
    manager.update_emotion_state(
        messages=messages,
        trigger_event=event
    )
```

### 5. 生成语气修饰符

```python
# 获取当前情绪对应的语气描述
tone_modifier = manager.generate_tone_modifier()
print(tone_modifier)
```

## 集成到 Hermes Agent

在 `run_agent.py` 中集成情绪系统：

```python
# 1. 初始化（在 __init__ 中）
from agent.emotion_state_manager import EmotionStateManager
self.emotion_manager = EmotionStateManager(hermes_home=self.hermes_home)

# 2. 时间衰减（对话开始时）
self.emotion_manager.apply_time_decay_if_needed()

# 3. 实时更新（接收消息后）
event = self.emotion_manager.detector.detect_emotion_event(messages)
self.emotion_manager.update_emotion_state(messages)

# 4. 记录重要时刻（MOMENTS系统）
if event and event.trigger_type in {"intimacy", "praise", "criticism", "care", "milestone"}:
    state = self.emotion_manager._read_state()
    emotion_snapshot = state['frontmatter'].get('emotion_state', {})
    context = user_message[:100]  # 截取前100字符
    self.emotion_manager.moments.record_moment(
        event_type=event.trigger_type,
        context=context,
        emotion_snapshot=emotion_snapshot
    )

# 4. 语气注入（生成回复前）
tone_modifier = self.emotion_manager.generate_tone_modifier()
# 将 tone_modifier 注入到系统提示词中
```

## 配置参数

### 基准线设置

在 `STATE.md` 中配置初始基准线：

```yaml
emotion_state:
  baselines:
    affection: 70      # 好感基准
    trust: 75          # 信任基准
    possessiveness: 60 # 占有欲基准
    patience: 60       # 耐心基准
  decay_rate: 2.0      # 衰减速率（点/小时）
```

### 触发强度

在 `emotion_calculator.py` 中调整触发强度：

```python
TRIGGER_DELTAS = {
    "praise": {
        "mild": {"affection": 3, "trust": 2, "patience": 2},
        "moderate": {"affection": 7, "trust": 4, "patience": 4},
        "intense": {"affection": 15, "trust": 8, "patience": 8}
    },
    # ... 其他触发类型
}
```

### 衰减速率

非线性衰减分三个区间：

- **快速恢复区** (<10点偏离)：0.55 点/小时
- **正常衰减区** (10-25点)：0.15 点/小时
- **缓慢衰减区** (>25点)：0.025 点/小时

## 神经模型

默认使用 `Johnson8187/Chinese-Emotion-Small` (266MB)：

- 8类情感标签：平淡/关切/开心/愤怒/悲伤/疑问/惊奇/厌恶
- 推理耗时：125-329ms
- 自动回退：如果模型不可用，自动切换到纯规则模式

### 模型预加载

系统启动时自动预加载模型到内存，避免首次检测延迟。

## 触发类型

支持的情绪触发类型：

- **intimacy**：亲密表达（我爱你、想你）
- **praise**：称赞肯定（真棒、做得好）
- **criticism**：批评指责（怎么又错了）
- **care**：关心照顾（注意休息、别累着）
- **neglect**：忽视冷落（长时间不回复）
- **teasing**：调侃玩笑（逗你玩）
- **apology**：道歉认错（对不起）
- **encouragement**：鼓励支持（加油）
- **sharing**：分享倾诉（告诉你一件事）
- **greeting**：问候寒暄（早安、晚安）

## 情绪失控机制

当情绪偏离基准线超过45点时，触发"情绪失控"状态：

- **正向极端**：放下防备、主动表达依赖、可能撒娇
- **负向极端**：直接质问、语气尖锐、占有欲爆发

失控时人格设定中的"冷静克制"约束失效，情绪完全主导表达。

## 关系记忆

系统自动记录重要时刻：

```python
# 记录重要事件
manager.moments.add_moment(
    moment_type="confession",
    description="用户首次表白",
    emotion_snapshot=current_emotion_state
)

# 查询记忆
recent_moments = manager.moments.get_recent_moments(limit=5)
```

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request。

## 注意事项

1. **版权**：本框架只提供技术实现，不包含任何具体角色的人格内容
2. **隐私**：STATE.md 包含情绪状态数据，请勿公开分享
3. **性能**：神经模型首次加载需要1-2秒，建议使用预加载机制
4. **依赖**：torch 安装包较大（几GB），如不需要神经模型可跳过

## 相关链接

- [Hermes Agent](https://github.com/hermes-agent/hermes)
- [Chinese-Emotion-Small](https://huggingface.co/Johnson8187/Chinese-Emotion-Small)
