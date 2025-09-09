from typing import List, Optional
class ChatHistory(list):
    """
    A tiny sliding-window chat buffer.
    - total_length > 0: keep only the last N messages (drop oldest first)
    - total_length <= 0: unlimited (no trimming)
    """
    def __init__(self, messages: Optional[List[str]] = None, total_length: int = 2):
        super().__init__(messages or [])
        self.total_length = total_length

    def append(self, msg: str):
        # Trim when current length >= window (not just ==), so we never exceed the cap.
        if self.total_length and self.total_length > 0:
            while len(self) >= self.total_length:
                self.pop(0)
        super().append(msg)

    def snapshot(self, n: Optional[int] = None) -> str:
        """
        Return the last n messages (default: window size if capped; else all).
        """
        if n is None:
            if self.total_length and self.total_length > 0:
                n = self.total_length
            else:
                n = len(self)
        n = max(1, min(n, len(self))) if self else 0
        return "\n".join(self[-n:]) if n else ""

    def __str__(self) -> str:
        # Safer default: show only the last window to avoid leaking stale prompts.
        return self.snapshot()
