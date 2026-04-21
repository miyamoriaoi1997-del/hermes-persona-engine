# Hermes Agent 集成指南

本文档说明如何将 Persona Engine 集成到 Hermes Agent 中。

## 文件放置

将以下文件复制到 Hermes Agent 的 `agent/` 目录：

```bash
cp emotion_detector.py ~/.hermes/hermes-agent/agent/
cp emotion_calculator.py ~/.hermes/hermes-agent/agent/
cp emotion_state_manager.py ~/.hermes/hermes-agent/agent/
cp sentiment_analyzer.py ~/.hermes/hermes-agent/agent/
cp moments_manager.py ~/.hermes/hermes-agent/agent/
```

## 配置文件

1. 创建 `~/.hermes/SOUL.md`（参考 SOUL.template.md）
2. 创建 `~/.hermes/STATE.md`（参考 STATE.template.md）

## 集成步骤

### 1. 在 run_agent.py 中导入

```python
from agent.emotion_state_manager import EmotionStateManager
```

### 2. 初始化情绪管理器

在 `__init__` 方法中添加：

```python
def __init__(self, ...):
    # ... 其他初始化代码
    
    # 初始化情绪系统
    try:
        self.emotion_manager = EmotionStateManager(
            hermes_home=self.hermes_home,
            decay_rate=2.0,
            update_body=True
        )
        logger.info("Emotion system initialized")
    except Exception as e:
        logger.warning(f"Emotion system init failed: {e}")
        self.emotion_manager = None
```

### 3. 时间衰减（对话开始时）

在 `run_conversation` 方法开始处添加：

```python
def run_conversation(self, ...):
    # 应用时间衰减
    if self.emotion_manager:
        try:
            self.emotion_manager.apply_time_decay_if_needed()
        except Exception as e:
            logger.warning(f"Time decay failed: {e}")
    
    # ... 继续对话流程
```

### 4. 实时情绪更新（接收消息后）

在接收到用户消息后立即更新情绪：

```python
# 接收用户消息
user_message = ...
messages.append(user_message)

# 检测并更新情绪
if self.emotion_manager:
    try:
        event = self.emotion_manager.detector.detect_emotion_event(messages)
        if event:
            self.emotion_manager.update_emotion_state(
                messages=messages,
                trigger_event=event
            )
    except Exception as e:
        logger.warning(f"Emotion update failed: {e}")
```

### 5. 语气注入（生成回复前）

在构建系统提示词时注入情绪修饰符：

```python
# 生成语气修饰符
tone_modifier = ""
if self.emotion_manager:
    try:
        tone_modifier = self.emotion_manager.generate_tone_modifier()
    except Exception as e:
        logger.warning(f"Tone generation failed: {e}")

# 构建系统提示词
system_prompt = f"""
{soul_content}

{tone_modifier}

{other_instructions}
"""
```

## 完整示例

```python
class HermesAgent:
    def __init__(self, hermes_home):
        self.hermes_home = Path(hermes_home)
        
        # 初始化情绪系统
        self.emotion_manager = EmotionStateManager(
            hermes_home=self.hermes_home
        )
    
    def run_conversation(self, messages):
        # 1. 时间衰减
        self.emotion_manager.apply_time_decay_if_needed()
        
        # 2. 接收用户消息
        user_input = input("User: ")
        messages.append({"role": "user", "content": user_input})
        
        # 3. 检测并更新情绪
        event = self.emotion_manager.detector.detect_emotion_event(messages)
        if event:
            self.emotion_manager.update_emotion_state(
                messages=messages,
                trigger_event=event
            )
        
        # 4. 生成语气修饰符
        tone_modifier = self.emotion_manager.generate_tone_modifier()
        
        # 5. 构建系统提示词
        soul_content = self.load_soul()
        system_prompt = f"{soul_content}\n\n{tone_modifier}"
        
        # 6. 调用 LLM 生成回复
        response = self.call_llm(system_prompt, messages)
        
        return response
```

## 注意事项

1. **异常处理**：所有情绪系统调用都应包裹在 try-except 中，避免影响主流程
2. **性能优化**：神经模型已在初始化时预加载，首次检测无延迟
3. **日志记录**：建议记录情绪触发事件，便于调试
4. **状态持久化**：情绪状态自动保存到 STATE.md，无需手动处理

## 调试

查看当前情绪状态：

```bash
cat ~/.hermes/STATE.md
```

测试情绪检测：

```python
from agent.emotion_state_manager import EmotionStateManager

manager = EmotionStateManager(hermes_home='/path/to/.hermes')
messages = [{'role': 'user', 'content': '你真棒！'}]
event = manager.detector.detect_emotion_event(messages)

if event:
    print(f"触发: {event.trigger_type}")
    print(f"变化: {event.deltas}")
```

## 性能指标

- 情绪检测耗时：125-329ms（含神经模型推理）
- 状态更新耗时：<10ms
- 内存占用：+266MB（神经模型）
- 首次加载：1-2秒（预加载后无延迟）
