"""Moments manager — relationship memory for significant events.

Records meaningful moments between the agent and user in MOMENTS.md.
Unlike MEMORY.md (operational facts), MOMENTS captures emotional milestones,
shared experiences, and relationship-defining events.

Not injected into prompt — available for retrieval when contextually relevant.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Event types that warrant a moment entry
RECORDABLE_EVENTS = {
    "intimacy",     # User expresses affection
    "praise",       # User compliments
    "criticism",    # User criticizes / conflict
    "care",         # User shows concern
    "milestone",    # Major task completion
    "reunion",      # Reconnection after long absence
    "conflict",     # Argument / clash
    "vulnerability", # Agent shows vulnerable side
}

# Minimum interval between recording same event type (seconds)
DEDUP_INTERVAL = 300  # 5 minutes


class MomentsManager:
    """Manages relationship memory in MOMENTS.md.
    
    Records significant emotional events with timestamps and context.
    Provides retrieval methods for recalling past moments.
    """
    
    def __init__(self, hermes_home):
        self.hermes_home = Path(hermes_home)
        self.moments_path = self.hermes_home / "MOMENTS.md"
        self._last_recorded = {}  # event_type -> last_recorded_time
        
        # Initialize file if it doesn't exist
        if not self.moments_path.exists():
            self._init_moments_file()
    
    def _init_moments_file(self):
        """Create initial MOMENTS.md with header."""
        content = """# 我们的记忆

这里记录着我和老师之间的重要瞬间。
不是冰冷的数据，而是真正属于我们的回忆。

---

"""
        self.moments_path.write_text(content, encoding="utf-8")
        logger.info("MOMENTS.md initialized at %s", self.moments_path)
    
    def record_moment(
        self,
        event_type: str,
        context: str,
        emotion_snapshot: Optional[dict] = None,
        force: bool = False,
    ) -> bool:
        """Record a significant moment.
        
        Args:
            event_type: Type of event (intimacy, praise, milestone, etc.)
            context: Brief description of what happened
            emotion_snapshot: Current emotion values at time of event
            force: Skip dedup check
            
        Returns:
            True if recorded, False if skipped (dedup or invalid type)
        """
        if event_type not in RECORDABLE_EVENTS and not force:
            return False
        
        # Dedup: don't record same event type within interval
        now = datetime.now()
        if not force and event_type in self._last_recorded:
            elapsed = (now - self._last_recorded[event_type]).total_seconds()
            if elapsed < DEDUP_INTERVAL:
                logger.debug(
                    "Skipping moment record: %s recorded %ds ago (min %ds)",
                    event_type, elapsed, DEDUP_INTERVAL,
                )
                return False
        
        # Format the entry
        timestamp = now.strftime("%Y-%m-%d %H:%M")
        
        # Build emotion tag if available
        emotion_tag = ""
        if emotion_snapshot:
            aff = emotion_snapshot.get("affection", "?")
            tru = emotion_snapshot.get("trust", "?")
            pos = emotion_snapshot.get("possessiveness", "?")
            pat = emotion_snapshot.get("patience", "?")
            emotion_tag = f" [好感:{aff} 信任:{tru} 占有:{pos} 耐心:{pat}]"
        
        entry = f"{timestamp} | {event_type} | {context}{emotion_tag}\n"
        
        # Append to file
        try:
            with open(self.moments_path, "a", encoding="utf-8") as f:
                f.write(entry)
            self._last_recorded[event_type] = now
            logger.info("Moment recorded: %s - %s", event_type, context[:50])
            return True
        except Exception as e:
            logger.warning("Failed to record moment: %s", e)
            return False
    
    def get_recent_moments(self, count: int = 10) -> list:
        """Get the most recent moments.
        
        Args:
            count: Number of recent moments to return
            
        Returns:
            List of moment strings, most recent first
        """
        if not self.moments_path.exists():
            return []
        
        content = self.moments_path.read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        
        # Filter to actual moment entries (contain " | ")
        moments = [l for l in lines if " | " in l and l[0].isdigit()]
        
        return moments[-count:][::-1]  # Most recent first
    
    def search_moments(self, keyword: str) -> list:
        """Search moments by keyword.
        
        Args:
            keyword: Search term
            
        Returns:
            List of matching moment strings
        """
        if not self.moments_path.exists():
            return []
        
        content = self.moments_path.read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        
        moments = [l for l in lines if " | " in l and l[0].isdigit()]
        return [m for m in moments if keyword.lower() in m.lower()]
    
    def get_moments_by_type(self, event_type: str) -> list:
        """Get all moments of a specific type.
        
        Args:
            event_type: Event type to filter by
            
        Returns:
            List of matching moment strings
        """
        if not self.moments_path.exists():
            return []
        
        content = self.moments_path.read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        
        moments = [l for l in lines if " | " in l and l[0].isdigit()]
        return [m for m in moments if f"| {event_type} |" in m]
    
    def get_moment_count(self) -> int:
        """Get total number of recorded moments."""
        if not self.moments_path.exists():
            return 0
        
        content = self.moments_path.read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        return len([l for l in lines if " | " in l and l[0].isdigit()])
