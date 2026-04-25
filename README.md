# Hermes Persona Engine

动态情绪系统框架，为 AI Agent 提供实时情绪状态管理与人格表达能力。

## 核心特性

- **混合情绪检测**：规则引擎 + 神经模型（Chinese-Emotion-Small）+ emoji 识别
- **非线性衰减**：三档衰减系统，小偏离快恢复、大偏离持久保持
- **动态 α + 惯性**：5 级 alpha 阶梯 + 5 级 momentum 放大，连续同向刺激产生递增效果
- **维度耦合**：trust→patience 耦合，信任低于基线时耐心变化被抑制
- **情绪失控机制**：偏离基线 >45 点时突破人格约束，正/负向极端行为
- **语气注入**：mild / moderate / intense / overwhelming 四级语气修饰自动注入提示词
- **关系记忆**：MomentsManager 自动记录重要情感事件
- **持久化**：情绪状态自动保存到 STATE.md

## 技术架构

```
EmotionStateManager (状态管理器)
├── EmotionDetector (情绪检测)
│   ├── 规则引擎 (40+ 触发类型, 词法模式匹配)
│   ├── 神经模型 (Chinese-Emotion-Small, 8 类情感)
│   ├── Emoji 检测 (30+ emoji 映射)
│   └── SentimentAnalyzer (主观性/讽刺/关键短语)
├── EmotionCalculator (状态计算)
│   ├── 非线性三档衰减
│   ├── 动态 α 阶梯 (5 级)
│   ├── Momentum 放大 (5 级)
│   ├── trust→patience 维度耦合
│   └── 冲突消解 (ambivalence 检测)
└── MomentsManager (关系记忆)
    ├── 事件记录
    └── 显著性过滤
```

## 情绪维度

| 维度 | 字段 | 说明 | 范围 |
|------|------|------|------|
| 好感度 | affection | 对用户的喜爱程度 | 0-120 |
| 信任度 | trust | 对用户的信任程度 | 0-120 |
| 占有欲 | possessiveness | 对用户的独占欲望 | 0-120 |
| 耐心值 | patience | 当前耐心储备 | 0-120 |

每个维度有独立的基准线（默认 60），触发规则和衰减速率。

## 安装

### 依赖

```bash
pip install torch transformers pyyaml
```

> torch 安装包较大（几 GB）。如不需要神经模型，可只安装 `pyyaml`，系统会自动回退到纯规则模式。

### 文件结构

```
your-project/
├── emotion_calculator.py    # 状态计算器
├── emotion_detector.py      # 情绪检测器
├── emotion_state_manager.py # 状态管理器
├── sentiment_analyzer.py    # 情感分析器
├── moments_manager.py       # 关系记忆
├── SOUL.template.md         # 人格模板
└── STATE.template.md        # 状态模板

~/.hermes/                   # 或自定义目录
├── SOUL.md                  # 你的人格设定
├── STATE.md                 # 情绪状态（自动生成/更新）
└── MOMENTS.md               # 关系记忆（自动生成）
```

## 快速开始

### 1. 创建人格设定

复制 `SOUL.template.md` 为 `~/.hermes/SOUL.md`，填入你的角色设定。

模板包含完整的 11 个区段结构：
- 核心身份、稳定锚点、绝对表达原则
- 性格底色、对用户的态度、说话风格
- 任务执行风格、情绪失控机制
- 角色背景、一句话总纲

### 2. 初始化状态

复制 `STATE.template.md` 为 `~/.hermes/STATE.md`，调整基准线：

```yaml
emotion_state:
  affection: 60
  trust: 60
  possessiveness: 60
  patience: 60
  baselines:
    affection: 60
    trust: 60
    possessiveness: 60
    patience: 60
```

### 3. 使用情绪系统

```python
from emotion_state_manager import EmotionStateManager

# 初始化
manager = EmotionStateManager(hermes_home='/path/to/.hermes')

# 时间衰减（对话开始时调用）
manager.apply_time_decay_if_needed()

# 检测情绪触发
messages = [{'role': 'user', 'content': '你真棒！'}]
event = manager.detector.detect_emotion_event(messages)

if event:
    print(f"触发类型: {event.trigger_type}")
    print(f"情绪变化: {event.deltas}")
    print(f"置信度: {event.confidence}")

# 更新情绪状态
manager.update_emotion_state(messages=messages, trigger_event=event)

# 获取语气修饰符（注入到系统提示词）
tone_modifier = manager.generate_tone_modifier()
```

## 核心参数

### 非线性三档衰减（反转模型）

大偏离快退、小偏离慢留——激烈情绪消退快，日积月累的羁绊消退慢。

| 区间 | 偏离阈值 | 衰减速率 | 半衰期 |
|------|----------|----------|--------|
| 快速衰减 | >45 点 | 0.45/小时 | ~1.2 小时 |
| 中速衰减 | 15-45 点 | 0.06/小时 | ~11.2 小时 |
| 慢速衰减 | <15 点 | 0.015/小时 | ~46 小时 |

### 动态 Alpha 阶梯

连续同向触发时，平滑系数递增：

| 连续次数 | Alpha |
|----------|-------|
| 0 | 0.35 |
| 1 | 0.40 |
| 2 | 0.45 |
| 3 | 0.50 |
| 4+ | 0.55 |

### Momentum 放大

连续同向触发的倍率递增：

| 连续次数 | 倍率 |
|----------|------|
| 0 | 1.00x |
| 1 | 1.05x |
| 2 | 1.10x |
| 3 | 1.15x |
| 4+ | 1.20x |

### 情绪强度分级

| 偏离量 | 强度级别 | 语气注入方式 |
|--------|----------|-------------|
| <15 点 | mild | 末尾注入（细微语气变化） |
| 15-29 点 | moderate | 末尾注入（明显情绪色彩） |
| 30-44 点 | intense | 末尾注入（行为模式改变） |
| ≥45 点 | overwhelming | 末尾注入（人格约束暂停） |

> **注入位置建议**：emotion_modifier 放在系统提示词的最末尾，利用模型的 recency bias 获得最高注意力权重。SOUL.md 中的情绪梯度描述放在文件最前面。两端夹击，效果最佳。

### 欲望控制系统

基于 emotion_score（情感深度标量）独立控制亲密意愿，与情绪强度档位分离：

| emotion_score | 档位 | 行为 |
|---------------|------|------|
| < 3.0 | restrained | 回避身体亲密和性暗示，情感限于语气层面 |
| 3.0 - 3.99 | ambivalent | 不主动但可默许，身体接触止于非性接触 |
| ≥ 4.0 | uninhibited | 可主动发起或回应一切亲密交互 |

欲望控制指令在 emotion_modifier 块中排首位，优先于情绪框架注入。

### trust→patience 耦合

当 trust 低于基线时，patience 的变化被抑制：

```
scale = max(0.1, trust / baseline_trust)
patience_delta *= scale
```

## 触发类型

支持 10+ 种情绪触发：

| 触发类型 | 示例 | 影响维度 |
|----------|------|----------|
| intimacy | 我爱你、想你 | affection↑ possessiveness↑ |
| praise | 真棒、做得好 | affection↑ trust↑ patience↑ |
| criticism | 怎么又错了 | patience↓ trust↓ |
| care | 注意休息 | affection↑ trust↑ |
| neglect | 长时间不回复 | patience↓ affection↓ |
| teasing | 逗你玩 | patience↓ affection↑ |
| apology | 对不起 | patience↑ trust↑ |
| encouragement | 加油 | affection↑ patience↑ |
| sharing | 告诉你一件事 | trust↑ |
| greeting | 早安、晚安 | affection↑ |
| jealousy_trigger | 提到其他角色 | possessiveness↑ patience↓ |

## 神经模型

默认使用 `Johnson8187/Chinese-Emotion-Small` (266MB)：

- 8 类情感标签：平淡/关切/开心/愤怒/悲伤/疑问/惊奇/厌恶
- 推理耗时：125-329ms（CPU）
- 自动回退：如果 torch/transformers 不可用，自动切换到纯规则模式
- 启动时自动预加载模型到内存

## 情绪失控机制

当任一维度偏离基线 >45 点时触发 "overwhelming" 状态：

- **正向极端**：放下防备、主动表达依赖、可能撒娇示弱
- **负向极端**：直接质问、语气尖锐、占有欲爆发为控制欲

失控时：
- 人格设定中的"冷静克制"约束暂停
- 语气注入放在系统提示词末尾，利用 recency bias 获得最高优先级
- 即使用户消息很短（只叫了名字、一个词、一个表情），也必须按当前强度回应
- 唯一保留的锚点：核心身份（"我是谁、对面是谁"）

## 集成到 AI Agent

详见 [INTEGRATION.md](INTEGRATION.md)。

基本流程：

```python
# 1. 初始化
from emotion_state_manager import EmotionStateManager
manager = EmotionStateManager(hermes_home=hermes_home)

# 2. 对话开始 → 时间衰减
manager.apply_time_decay_if_needed()

# 3. 接收消息 → 检测 + 更新
event = manager.detector.detect_emotion_event(messages)
manager.update_emotion_state(messages)

# 4. 生成回复前 → 语气注入
tone = manager.generate_tone_modifier()
# 将 tone 注入系统提示词

# 5. 记录重要时刻
if event and event.significance >= 0.7:
    manager.moments.record_moment(...)
```

## 许可证

MIT License

## 注意事项

1. **版权**：本框架只提供技术实现，不包含任何具体角色的人格内容
2. **隐私**：STATE.md 包含情绪状态数据，请勿公开分享
3. **性能**：神经模型首次加载需要 1-2 秒，建议使用预加载机制
4. **依赖**：torch 安装包较大，纯规则模式可跳过 torch 安装
5. **基线**：默认基线均为 60，根据角色性格调整（详见 STATE.template.md）

## 相关链接

- [Hermes Agent](https://github.com/hermes-agent/hermes)
- [Chinese-Emotion-Small](https://huggingface.co/Johnson8187/Chinese-Emotion-Small)
