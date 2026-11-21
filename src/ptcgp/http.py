"""Client HTTP minimal (bibliothèque standard) qui respecte la limite de requêtes de Limitless.

Sans clé, l'API Limitless accepte 50 requêtes par fenêtre de 5 minutes. Chaque réponse porte un en-tête
``RateLimit: "50-in-5min"; r=49; t=300`` (``r`` : requêtes restantes, ``t`` : secondes avant la remise
à zéro). Le client s'arrête quand ``r`` tombe à 0 et reprend à la fenêtre suivante.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from typing import Any, Callable

from .config import USER_AGENT

_RATELIMIT = re.compile(r"\br=(\d+)\s*;\s*t=(\d+)")


def parse_ratelimit(value: str | None) -> tuple[int, int] | None:
    """Renvoie (requêtes restantes, secondes avant remise à zéro), ou None si l'en-tête est absent."""
    if not value:
        return None
    match = _RATELIMIT.search(value)
    return (int(match.group(1)), int(match.group(2))) if match else None


class HttpClient:
    def __init__(
        self,
        min_interval: float = 0.0,
        max_retries: int = 5,
        log: Callable[[str], None] = print,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.log = log
        self._sleep = sleep
        self._clock = clock
        self._last_request: float | None = None
        self.remaining: int | None = None
        self.reset_at: float | None = None
        self.requests = 0

    def _wait_turn(self) -> None:
        now = self._clock()
        if self.remaining is not None and self.remaining <= 0 and self.reset_at is not None and now < self.reset_at:
            wait = self.reset_at - now + 1
            self.log(f"Limite de requêtes atteinte : pause de {wait:.0f} s")
            self._sleep(wait)
            self.remaining = None
        if self._last_request is not None:
            gap = self._clock() - self._last_request
            if gap < self.min_interval:
                self._sleep(self.min_interval - gap)

    def _note_limits(self, headers: Any) -> None:
        limits = parse_ratelimit(headers.get("RateLimit") if headers is not None else None)
        if limits:
            self.remaining, reset_in = limits
            self.reset_at = self._clock() + reset_in

    def _retry_delay(self, headers: Any, attempt: int) -> float:
        retry_after = headers.get("Retry-After") if headers is not None else None
        if retry_after and retry_after.isdigit():
            return int(retry_after) + 1
        if self.remaining == 0 and self.reset_at is not None:
            return max(self.reset_at - self._clock(), 1) + 1
        return min(5 * 2 ** (attempt - 1), 120)

    def get_bytes(self, url: str) -> bytes:
        for attempt in range(1, self.max_retries + 1):
            self._wait_turn()
            self._last_request = self._clock()
            self.requests += 1
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    self._note_limits(response.headers)
                    return response.read()
            except urllib.error.HTTPError as err:
                self._note_limits(err.headers)
                if err.code != 429 and err.code < 500:
                    raise
                wait = self._retry_delay(err.headers, attempt)
                self.log(f"HTTP {err.code} sur {url} : nouvel essai dans {wait:.0f} s ({attempt}/{self.max_retries})")
            except (urllib.error.URLError, TimeoutError) as err:
                wait = min(5 * 2 ** (attempt - 1), 120)
                self.log(f"Erreur réseau sur {url} ({err}) : nouvel essai dans {wait:.0f} s ({attempt}/{self.max_retries})")
            self._sleep(wait)
        raise RuntimeError(f"Échec après {self.max_retries} essais : {url}")

    def get_json(self, url: str) -> Any:
        return json.loads(self.get_bytes(url).decode("utf-8"))
