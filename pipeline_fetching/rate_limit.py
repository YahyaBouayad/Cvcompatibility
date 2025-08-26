import time
import threading
from typing import Optional

class TokenBucket:
    """Simple rate limiter thread-safe."""
    def __init__(self, rate_per_sec: float, capacity: Optional[float] = None):
        self.rate = float(rate_per_sec)
        self.capacity = capacity if capacity is not None else max(1.0, self.rate * 2)
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, tokens: float = 1.0):
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last
                self._last = now
                self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
            time.sleep(max(0.01, tokens / self.rate / 2))
 