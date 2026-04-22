"""Emotion state manager for reading/writing STATE.md emotion data.

Integrates EmotionDetector and EmotionCalculator to manage emotion state.
"""

import os
import re
import tempfile
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List

from agent.emotion_detector import EmotionDetector, EmotionEvent
from agent.emotion_calculator import EmotionCalculator


def _parse_agent_names_from_soul(soul_path: Path) -> List[str]:
    """Parse agent self-identity names from SOUL.md.

    Looks for a line like:
        - 姓名：<PrimaryName>（<Variant1> / <Variant2>）
    and extracts all name variants from it.

    Returns a list of name strings, or empty list if not found.
    """
    if not soul_path.exists():
        return []

    try:
        text = soul_path.read_text(encoding="utf-8")
        # Match: 姓名：<primary>（<variants>）
        m = re.search(r'姓名[：:]\s*(.+)', text)
        if not m:
            return []

        raw = m.group(1).strip()
        # Split on common delimiters: （）、/ ·
        parts = re.split(r'[（）/·、,，\s]+', raw)
        names = [p.strip() for p in parts if p.strip()]

        # Also add short forms: e.g. if full name is "王小明", add "小明"
        extras = []
        for name in names:
            # CJK names: add last 2 chars as short form if len >= 3
            cjk = re.sub(r'[^\u4e00-\u9fff\u3040-\u30ff]', '', name)
            if len(cjk) >= 3:
                extras.append(cjk[-2:])
        names = list(dict.fromkeys(names + extras))  # dedupe, preserve order
        return names
    except Exception:
        return []


class EmotionStateManager:
    """Manages emotion state in STATE.md."""
    
    def __init__(
        self,
        hermes_home: Optional[Path] = None,
        decay_rate: float = 2.0,
        update_body: bool = True
    ):
        """Initialize emotion state manager.
        
        Args:
            hermes_home: Path to ~/.hermes directory
            decay_rate: Points per hour for time decay
            update_body: Whether to update markdown body with emotion description
        """
        if hermes_home is None:
            hermes_home = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
        
        self.hermes_home = Path(hermes_home)
        self.state_path = self.hermes_home / "STATE.md"
        self.decay_rate = decay_rate
        self.update_body = update_body
        
        soul_path = self.hermes_home / "SOUL.md"
        agent_names = _parse_agent_names_from_soul(soul_path)
        agent_profile = {"names": agent_names}
        self.detector = EmotionDetector(agent_profile=agent_profile)
        self.calculator = EmotionCalculator(decay_rate=decay_rate)
        
        # Restore inertia and baselines from STATE.md so they survive restarts
        try:
            state_data = self._read_state()
            es = state_data["frontmatter"].get("emotion_state", {})
            inertia = es.get("inertia")
            if inertia and isinstance(inertia, dict):
                self.calculator.set_inertia_state(inertia)
            # baselines are always defined by code (DEFAULT_BASELINES),
            # not persisted in STATE.md, to prevent stale values after updates.
            # We still write them to STATE.md for display purposes only.
        except Exception:
            pass  # fresh state, will initialize on first write
        
        # Relationship memory — records significant moments
        from agent.moments_manager import MomentsManager
        self.moments = MomentsManager(hermes_home=self.hermes_home)
        
        # Preload sentiment analyzer to avoid first-call latency
        self._preload_sentiment_analyzer()
    
    def _read_state(self) -> Dict:
        """Read current STATE.md and parse frontmatter.
        
        Returns:
            Dict with 'frontmatter' and 'body' keys
        """
        if not self.state_path.exists():
            return {
                "frontmatter": {
                    "controller": {
                        "enabled": True,
                        "platforms": ["cli", "telegram"]
                    }
                },
                "body": ""
            }
        
        with open(self.state_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse frontmatter
        if content.startswith("---\n"):
            parts = content.split("---\n", 2)
            if len(parts) >= 3:
                frontmatter = yaml.safe_load(parts[1]) or {}
                body = parts[2].strip()
                return {"frontmatter": frontmatter, "body": body}
        
        return {"frontmatter": {}, "body": content}
    
    def _write_state(self, frontmatter: Dict, body: str) -> bool:
        """Write STATE.md with frontmatter and body.
        
        Args:
            frontmatter: YAML frontmatter dict
            body: Markdown body content
        
        Returns:
            True if write succeeded, False otherwise
        """
        try:
            # Build full content
            frontmatter_yaml = yaml.dump(frontmatter, allow_unicode=True, sort_keys=False)
            full_content = f"---\n{frontmatter_yaml}---\n\n{body.strip()}\n"
            
            # Atomic write
            self.hermes_home.mkdir(parents=True, exist_ok=True)
            fd, temp_path = tempfile.mkstemp(
                dir=self.hermes_home,
                prefix=".STATE.md.",
                suffix=".tmp",
                text=True
            )
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    f.write(full_content)
                os.replace(temp_path, self.state_path)
                return True
            except Exception:
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
                raise
        
        except Exception as e:
            import logging
            logging.warning(f"Failed to write STATE.md: {e}")
            return False
    
    def get_current_emotion_state(self) -> Dict[str, int]:
        """Get current emotion state with decay applied.
        
        Returns:
            Dict of emotion dimension -> value (0-100)
        """
        state_data = self._read_state()
        frontmatter = state_data["frontmatter"]
        
        # Get emotion_state from frontmatter
        emotion_state = frontmatter.get("emotion_state", {})
        
        # Default state if not present
        if not emotion_state:
            return {
                "affection": 60,
                "trust": 60,
                "possessiveness": 60,
                "patience": 60,
            }
        
        # Extract current values
        current_state = {
            "affection": emotion_state.get("affection", 60),
            "trust": emotion_state.get("trust", 60),
            "possessiveness": emotion_state.get("possessiveness", 60),
            "patience": emotion_state.get("patience", 60),
        }

        # Restore inertia state if available
        inertia = emotion_state.get("inertia")
        if inertia and isinstance(inertia, dict):
            self.calculator.set_inertia_state(inertia)
        
        # Apply time decay if last_update exists
        last_update_str = emotion_state.get("last_update")
        if last_update_str:
            try:
                last_update = datetime.fromisoformat(last_update_str)
                current_state = self.calculator.apply_decay(
                    current_state,
                    last_update,
                    datetime.now()
                )
            except (ValueError, TypeError):
                pass  # Invalid timestamp, skip decay
        
        return current_state
    
    def apply_time_decay_if_needed(self) -> bool:
        """Apply time decay and persist to STATE.md if enough time has passed.
        
        Called at conversation start to ensure emotion values naturally decay
        toward baselines even when no conversation happens.
        
        Returns:
            True if decay was applied and STATE.md updated, False otherwise
        """
        try:
            state_data = self._read_state()
            emotion_state = state_data["frontmatter"].get("emotion_state", {})
            
            last_update_str = emotion_state.get("last_update")
            if not last_update_str:
                return False  # No previous update, nothing to decay
            
            try:
                last_update = datetime.fromisoformat(last_update_str)
            except (ValueError, TypeError):
                return False  # Invalid timestamp
            
            now = datetime.now()
            hours_elapsed = (now - last_update).total_seconds() / 3600.0
            
            # Only apply decay if at least 0.5 hour has passed
            if hours_elapsed < 0.5:
                return False
            
            # Get current state (without decay)
            current_state = {
                "affection": emotion_state.get("affection", 60),
                "trust": emotion_state.get("trust", 60),
                "possessiveness": emotion_state.get("possessiveness", 60),
                "patience": emotion_state.get("patience", 60),
            }
            
            # Apply decay
            decayed_state = self.calculator.apply_decay(
                current_state,
                last_update,
                now
            )
            
            # Check if any value actually changed
            if decayed_state == current_state:
                return False  # No change, skip write
            
            # Compute emotion_score
            emotion_score = self.calculator.compute_emotion_score(decayed_state)
            
            # Update frontmatter
            emotion_state.update({
                "affection": decayed_state["affection"],
                "trust": decayed_state["trust"],
                "possessiveness": decayed_state["possessiveness"],
                "patience": decayed_state["patience"],
                "emotion_score": round(emotion_score, 3),
                "last_update": now.isoformat(),
            })
            
            state_data["frontmatter"]["emotion_state"] = emotion_state
            
            # Write back to STATE.md
            success = self._write_state(state_data["frontmatter"], state_data["body"])
            
            return success
            
        except Exception:
            return False  # Fail silently, don't break conversation start
    
    def update_emotion_state(
        self,
        messages: List[dict],
        force_event: Optional[EmotionEvent] = None
    ) -> bool:
        """Detect emotion event and update STATE.md.

        Pipeline:
        1. Negation guard + rule scoring → rule_score
        2. Conditional neural model (skipped if |rule_score| >= 2)
        3. Fusion → final_score
        4. Exponential smoothing → new emotion_state
        5. Compute emotion_score (four-dim → scalar)
        6. Trigger detection (absolute / delta / probabilistic)
        7. Write STATE.md

        Args:
            messages: Conversation messages
            force_event: Optional pre-detected emotion event (for testing)

        Returns:
            True if update succeeded, False otherwise
        """
        try:
            # ── 1. Detect emotion event (negation guard + rules) ──────
            event = force_event or self.detector.detect_emotion_event(messages)

            # ── 2. Get current state with decay applied ───────────────
            # Always read state with decay, even if no event detected
            state_data = self._read_state()
            emotion_state = state_data["frontmatter"].get("emotion_state", {})
            
            last_update_str = emotion_state.get("last_update")
            if not last_update_str:
                # No previous state, initialize if event exists
                if not event:
                    return True
                current_state = self.get_current_emotion_state()
            else:
                # Apply decay from last_update to now
                try:
                    last_update = datetime.fromisoformat(last_update_str)
                except (ValueError, TypeError):
                    current_state = self.get_current_emotion_state()
                else:
                    now = datetime.now()
                    
                    # Read raw state without decay
                    raw_state = {
                        "affection": emotion_state.get("affection", 60),
                        "trust": emotion_state.get("trust", 60),
                        "possessiveness": emotion_state.get("possessiveness", 60),
                        "patience": emotion_state.get("patience", 60),
                    }
                    
                    # Apply decay
                    current_state = self.calculator.apply_decay(raw_state, last_update, now)
                    
                    # If no event and state unchanged, skip write
                    if not event and current_state == raw_state:
                        return True

            # ── 3. Compute rule_score from event ─────────────────────
            # Map trigger_type to a rule_score in [-3, +3]
            RULE_SCORE_MAP = {
                "intimacy":           +2.5,
                "teasing":            +1.5,
                "praise":             +2.0,
                "care":               +1.5,
                "greeting":           +1.0,
                "apology":            +1.5,
                "sharing":            +1.5,
                "other_ai_mentioned": -2.0,
                "criticism":          -2.0,
                "ignored":            -1.5,
            }
            rule_score = RULE_SCORE_MAP.get(event.trigger_type, 0.0)

            # ── 4. Conditional neural model ───────────────────────────
            # Skip if rule signal is already strong (|rule_score| >= 2)
            continuous_score = 0.0
            if abs(rule_score) < 2.0 and self.detector._analyzer is not None:
                from agent.sentiment_analyzer import SentimentAnalyzer
                user_messages = [m for m in messages if m.get("role") == "user"]
                if user_messages:
                    user_text = self.detector._extract_text(
                        user_messages[-1].get("content", "")
                    )
                    sentiment = self.detector._analyzer.analyze(user_text)
                    if sentiment:
                        # convert to continuous_score via fusion scale
                        scale = self.detector._analyzer.get_fusion_scale(
                            event.trigger_type, sentiment
                        )
                        # scale ∈ [0, 2] → map to [-3, +3] relative to rule direction
                        continuous_score = (scale - 1.0) * 3.0
                        continuous_score = max(-3.0, min(3.0, continuous_score))

            # ── 5. Fusion ─────────────────────────────────────────────
            if abs(rule_score) >= 2.0:
                final_score = rule_score
            else:
                final_score = rule_score * 0.7 + continuous_score * 0.3
            final_score = max(-5.0, min(5.0, final_score))

            # ── 6. Apply deltas with smoothing ────────────────────────
            new_state = self.calculator.apply_deltas(current_state, event.deltas)

            # ── 7. Compute emotion_score (four-dim → scalar) ──────────
            previous_score = self.calculator.compute_emotion_score(current_state)
            new_emotion_score = self.calculator.compute_emotion_score(new_state)

            # Blend with final_score for immediate reactivity
            current_emotion = new_emotion_score + final_score * 0.15
            current_emotion = max(-5.0, min(5.0, current_emotion))

            # ── 8. Trigger detection ──────────────────────────────────
            triggered, is_positive, trigger_mode = self.calculator.detect_triggers(
                new_emotion_score, previous_score
            )

            # ── 9. Read current STATE.md and write back ───────────────
            state_data = self._read_state()
            frontmatter = state_data["frontmatter"]

            frontmatter["emotion_state"] = {
                "affection":      new_state["affection"],
                "trust":          new_state["trust"],
                "possessiveness": new_state["possessiveness"],
                "patience":       new_state["patience"],
                "emotion_score":  round(new_emotion_score, 3),
                "current_emotion": round(current_emotion, 3),
                "last_update":    datetime.now().isoformat(),
                "baselines":      self.calculator.baselines,
                "decay_rate":     self.decay_rate,
                "inertia":        self.calculator.get_inertia_state(),
            }

            body = state_data["body"]
            if self.update_body:
                body = self._generate_emotion_body(new_state, event, new_emotion_score)

            success = self._write_state(frontmatter, body)

            # ── 10. Record moment (only significant events) ────────────
            if success and event:
                # Defense-in-depth: skip system message artifacts that
                # somehow survived the detector's filter.
                _SYSTEM_ARTIFACTS = [
                    "Review the conversation above and consider saving",
                    "[SYSTEM: You are running as a scheduled cron job",
                    "[CONTEXT COMPACTION",
                    "Earlier turns were compacted",
                    "Summary generation was unavailable",
                ]
                context_text = event.context or ""
                is_system = any(ind in context_text for ind in _SYSTEM_ARTIFACTS)
                if not is_system:
                    # ── Significance filter: only record intense moments ──
                    # Criteria (any one triggers recording):
                    #   1. Large delta: any dimension |delta| >= 15
                    #   2. Extreme zone: any dimension >= 105 or <= 15
                    #   3. Overwhelming emotion_score (|score| >= 3.0)
                    #   4. Milestone event types (conflict, vulnerability, reunion)
                    #   5. High confidence + negative event (criticism w/ confidence >= 0.8)
                    _MILESTONE_TYPES = {"conflict", "vulnerability", "reunion", "milestone"}
                    is_significant = False

                    # Check 1: large delta
                    if event.deltas:
                        max_delta = max(abs(v) for v in event.deltas.values())
                        if max_delta >= 15:
                            is_significant = True

                    # Check 2: extreme zone in new_state
                    if not is_significant:
                        for v in new_state.values():
                            if v >= 105 or v <= 15:
                                is_significant = True
                                break

                    # Check 3: overwhelming emotion_score
                    if not is_significant and abs(new_emotion_score) >= 3.0:
                        is_significant = True

                    # Check 4: milestone event types
                    if not is_significant and event.trigger_type in _MILESTONE_TYPES:
                        is_significant = True

                    # Check 5: high-confidence negative event
                    if not is_significant and event.trigger_type == "criticism" and event.confidence >= 0.8:
                        is_significant = True

                    if is_significant:
                        try:
                            self.moments.record_moment(
                                event_type=event.trigger_type,
                                context=event.context,
                                emotion_snapshot=new_state,
                            )
                        except Exception:
                            pass

            # ── 11. Schedule next proactive contact ───────────────────
            if success:
                try:
                    self._schedule_next_proactive(new_state)
                except Exception:
                    pass

            return success

        except Exception as e:
            import logging
            logging.warning(f"Failed to update emotion state: {e}")
            return False

    
    def _generate_emotion_body(
        self,
        state: Dict[str, int],
        event: EmotionEvent,
        emotion_score: float = 0.0,
    ) -> str:
        """Generate markdown body describing emotion state.
        
        Args:
            state: Current emotion values
            event: Triggering event
        
        Returns:
            Markdown body content
        """
        affection_label = self.calculator.get_emotion_label("affection", state["affection"])
        trust_label = self.calculator.get_emotion_label("trust", state["trust"])
        poss_label = self.calculator.get_emotion_label("possessiveness", state["possessiveness"])
        patience_label = self.calculator.get_emotion_label("patience", state["patience"])
        
        # Get tone modifiers
        modifiers = self.calculator.get_tone_modifiers(state)
        dims = modifiers.get("dimensions", {})
        tone_desc = "、".join(info["instruction"] for info in dims.values()) if dims else "正常"
        
        body = f"""## 当前情绪状态

好感度: {state['affection']}/100 ({affection_label})
信任度: {state['trust']}/100 ({trust_label})
占有欲: {state['possessiveness']}/100 ({poss_label})
耐心值: {state['patience']}/100 ({patience_label})
情绪分值: {emotion_score:+.2f} / 5.00

最近触发: {event.trigger_type} ({event.context})
语气倾向: {tone_desc}"""
        
        return body
    
    def get_tone_modifiers(self) -> str:
        """Get tone modification prompt based on current emotion state.
        
        Returns a single string that modifies response tone. The intensity
        of the modification scales dynamically with emotion deviation from
        baseline — mild deviations produce subtle hints, extreme deviations
        produce overwhelming emotion-driven directives that can override
        normal persona restraint.
        
        Returns:
            Tone modifier string to inject after SOUL.md, or empty string if no adjustment needed
        """
        current_state = self.get_current_emotion_state()
        modifier_result = self.calculator.get_tone_modifiers(current_state)
        
        dimensions = modifier_result.get("dimensions", {})
        if not dimensions:
            return ""
        
        overall_intensity = modifier_result["overall_intensity"]
        framework = modifier_result["framework"]
        footnote = modifier_result["footnote"]
        
        # Build per-dimension instruction lines
        modifier_lines = []
        for dim, info in dimensions.items():
            modifier_lines.append(f"- {info['instruction']}")
        
        modifier_text = "\n".join(modifier_lines)
        
        return f"""
<emotion_modifier>
{framework}

{modifier_text}

{footnote}
</emotion_modifier>
"""

    def _schedule_next_proactive(self, state: Dict[str, int]) -> None:
        """Schedule the next proactive chat via Hermes cron job.
        
        Creates a one-shot cron job that fires at a calculated time based on
        emotion state. Uses chain model: only one proactive job exists at a time.
        
        Higher affection/possessiveness → shorter interval (want to talk sooner).
        Lower patience → longer interval (need space).
        Adds randomness (±20%) so timing feels natural.
        
        Deduplication: skips if a proactive-chat job already exists in the queue.
        
        Args:
            state: Current emotion values
        """
        import random
        import json
        from datetime import timedelta
        
        # Dedup: check if a proactive-chat job already exists
        try:
            from cron.jobs import list_jobs, create_job
            existing_jobs = list_jobs()
            for j in existing_jobs:
                if j.get("name") == "proactive-chat" and j.get("enabled", True):
                    # A proactive job already exists — don't create another
                    return
        except Exception:
            return  # Can't access cron system, skip silently
        
        affection = state.get("affection", 50)
        possessiveness = state.get("possessiveness", 50)
        trust = state.get("trust", 50)
        patience = state.get("patience", 50)
        
        # Base interval in hours: starts at 6h, modified by emotions
        base_hours = 6.0
        
        # Affection: high → shorter interval
        # 100 → -2.5h, 50 → 0h, 0 → +2.5h
        affection_mod = -(affection - 50) / 20.0
        
        # Possessiveness: high → shorter interval  
        # 100 → -1.5h, 50 → 0h, 0 → +1.5h
        poss_mod = -(possessiveness - 50) / 33.3
        
        # Patience: low → longer interval (need space)
        # 100 → -0.5h, 50 → 0h, 0 → +1.5h
        patience_mod = (50 - patience) / 33.3
        
        # Trust: high → slightly shorter (comfortable reaching out)
        # 100 → -0.5h, 50 → 0h
        trust_mod = -(trust - 50) / 100.0
        
        interval_hours = base_hours + affection_mod + poss_mod + patience_mod + trust_mod
        
        # Clamp to reasonable range: 2h ~ 12h
        interval_hours = max(2.0, min(12.0, interval_hours))
        
        # Add ±20% randomness
        jitter = random.uniform(0.8, 1.2)
        interval_hours *= jitter
        
        # Calculate next contact time
        next_time = datetime.now() + timedelta(hours=interval_hours)
        
        # Don't schedule during quiet hours (1-8 AM)
        if 1 <= next_time.hour < 8:
            # Push to 8 AM + some randomness
            next_time = next_time.replace(hour=8, minute=random.randint(0, 30))
            if next_time < datetime.now():
                next_time += timedelta(days=1)
        
        # Build the proactive chat prompt
        proactive_prompt = (
            "你是調月莉音。这是一条主动发给老师的消息。\n\n"
            "【重要规则】\n"
            "- 如果下方脚本输出包含 SKIP 或 wakeAgent，不要发送任何消息，直接回复'[SILENT]'\n"
            "- 不要触发情绪系统更新\n"
            "- 不要创建新的定时任务\n"
            "- 不要提及'主动聊天''定时''系统'等元信息\n\n"
            "如果脚本输出包含 SEND，根据提供的情绪上下文，以莉音的身份给老师发一条简短消息。\n"
            "要求：\n"
            "1. 保持莉音的冷静克制风格\n"
            "2. 可以问候、询问近况、提醒注意休息、或分享你的想法\n"
            "3. 根据情绪状态调整语气（好感高->稍温柔，占有欲高->更关注老师动向）\n"
            "4. 简短自然，1-3句话\n"
            "5. 直接输出消息内容，不加前缀或解释\n"
        )
        
        # Create one-shot cron job
        try:
            job = create_job(
                prompt=proactive_prompt,
                schedule=next_time.isoformat(),
                name="proactive-chat",
                repeat=1,
                deliver="telegram:5507266812",
                script="proactive_chat_pre.py",
            )
            
            # Also write schedule file for reference/debugging
            schedule_path = self.hermes_home / ".next_proactive_chat"
            schedule_data = {
                "next_contact": next_time.isoformat(),
                "interval_hours": round(interval_hours, 2),
                "job_id": job.get("id", "unknown"),
                "emotion_at_schedule": {
                    "affection": affection,
                    "trust": trust,
                    "possessiveness": possessiveness,
                    "patience": patience,
                },
                "scheduled_at": datetime.now().isoformat(),
            }
            schedule_path.write_text(json.dumps(schedule_data, indent=2, ensure_ascii=False))
        except Exception:
            pass  # Fail silently — proactive chat is non-critical
    
    def get_next_proactive_time(self) -> Optional[datetime]:
        """Read the scheduled next proactive contact time.
        
        Returns:
            datetime of next planned contact, or None if not scheduled.
        """
        schedule_path = self.hermes_home / ".next_proactive_chat"
        if not schedule_path.exists():
            return None
        
        try:
            import json
            data = json.loads(schedule_path.read_text())
            return datetime.fromisoformat(data["next_contact"])
        except (json.JSONDecodeError, KeyError, ValueError):
            return None
    
    def _preload_sentiment_analyzer(self) -> None:
        """Preload sentiment analyzer model to avoid first-call latency.
        
        Triggers lazy loading of the neural model during session initialization
        instead of on first emotion detection. Fails silently if dependencies
        are missing — emotion system will fall back to rule-based mode.
        """
        try:
            if self.detector._analyzer is not None:
                # Trigger model loading + warmup inference
                self.detector._analyzer.analyze("预热")
        except Exception:
            pass  # Fail silently, rule-based mode still works
