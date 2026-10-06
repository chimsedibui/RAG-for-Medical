"""HTTP plumbing for the crawler: retrying session, per-host throttle, robots.txt cache."""
import threading
import time
from collections import defaultdict
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/128.0.0.0 Safari/537.36 R2AI2026-research-crawler")


class Throttle:
    """Min interval between requests to the same host, across threads."""
    def __init__(self, delay: float):
        self.delay, self._next, self._lock = delay, defaultdict(float), threading.Lock()

    def wait(self, host: str) -> None:
        with self._lock:
            now = time.monotonic()
            at = max(now, self._next[host])
            self._next[host] = at + self.delay
        if at > now:
            time.sleep(at - now)


def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504],
                  respect_retry_after_header=True)
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=128, pool_maxsize=64))
    s.mount("http://", HTTPAdapter(max_retries=retry, pool_connections=128, pool_maxsize=64))
    s.headers.update({"User-Agent": UA, "Accept-Language": "vi,en;q=0.8,zh;q=0.6"})
    return s


class Robots:
    """robots.txt cache. Per-host lock so one slow host never blocks fetches for other hosts."""
    def __init__(self, sess: requests.Session, enabled: bool):
        self.enabled, self.sess = enabled, sess
        self._cache: dict[str, RobotFileParser] = {}
        self._locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._guard = threading.Lock()

    def allowed(self, url: str) -> bool:
        if not self.enabled:
            return True
        u = urlparse(url)
        base = f"{u.scheme}://{u.netloc}"
        with self._guard:
            lock = self._locks[base]
        with lock:
            rp = self._cache.get(base)
            if rp is None:
                rp = RobotFileParser()
                try:
                    r = self.sess.get(f"{base}/robots.txt", timeout=10)
                    rp.parse(r.text.splitlines() if r.status_code == 200 else [])
                except requests.RequestException:
                    rp.parse([])
                self._cache[base] = rp
        return rp.can_fetch(UA, url)
