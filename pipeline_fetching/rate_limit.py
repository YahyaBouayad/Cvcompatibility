import time
import threading

class TokenBucket:
    """
    Token bucket avec horloge monotonic + Condition.wait()
    - Pas d'attente occupée
    - Possibilité d'ajouter un timeout si besoin
    """
    def __init__(self, rate_per_sec: float, capacity: float = None):
        self.rate = float(rate_per_sec)
        self.capacity = capacity if capacity is not None else max(1.0, self.rate)
        self.tokens = self.capacity
        self.last = time.monotonic()
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)

    def _refill_unlocked(self):
        now = time.monotonic()
        delta = now - self.last
        if delta > 0:
            self.tokens = min(self.capacity, self.tokens + delta * self.rate)
            self.last = now

    def acquire(self, n: float = 1.0, timeout: float = None):
        """
        Bloque jusqu'à disposer de n jetons.
        timeout (sec) optionnel: lève TimeoutError si dépassé.
        """
        end = None if timeout is None else (time.monotonic() + timeout)
        with self.cv:
            while True:
                self._refill_unlocked()
                if self.tokens >= n:
                    self.tokens -= n
                    return
                # temps d'attente estimé jusqu'au prochain jeton
                need = max(n - self.tokens, 0.0)
                wait = max(need / self.rate, 0.01)
                if timeout is None:
                    self.cv.wait(timeout=wait)
                else:
                    remaining = end - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Rate limiter acquire timeout")
                    self.cv.wait(timeout=min(wait, max(remaining, 0.01)))
