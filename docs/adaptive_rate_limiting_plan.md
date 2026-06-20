# Adaptive Rate-Limiting Plan

## Context

The current `QanoonScraper` uses fixed `delay_min` / `delay_max` values and a configurable worker count. This works well when the target site has stable response behavior, but it cannot react to dynamic throttling or CAPTCHA deployment. This document outlines a plan for adding adaptive rate limiting without changing the existing production code path.

## Goals

1. Detect when the site is throttling or serving challenges.
2. Automatically reduce request pressure before a hard block occurs.
3. Restore throughput when the site returns to normal behavior.
4. Keep the change localized so it can be enabled via configuration.

## Proposed Design

### 1. Signals to Monitor

| Signal | Meaning | Action |
|---|---|---|
| HTTP 429 Too Many Requests | Explicit rate limit | Increase delays, reduce workers |
| HTTP 403 Forbidden | Possible block or challenge | Pause, increase delays, rotate identity |
| HTTP 5xx errors | Transient server stress | Exponential backoff |
| Response time spike | Soft throttling | Reduce concurrency |
| HTML contains "captcha", "challenge", "are you human" | CAPTCHA or JS challenge | Pause and alert / trigger solver |
| Empty responses or size drop | Bot detection response | Retry with different identity |

### 2. Adaptive Delay Controller

Introduce an `AdaptiveDelayController` class that maintains a current `base_delay` value.

```python
class AdaptiveDelayController:
    def __init__(self, min_delay=0.2, max_delay=30.0, backoff_factor=2.0, recovery_factor=0.9):
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.current_delay = min_delay
        self.backoff_factor = backoff_factor
        self.recovery_factor = recovery_factor

    def on_success(self, response_time: float):
        # Slowly reduce delay after healthy responses
        self.current_delay = max(self.min_delay, self.current_delay * self.recovery_factor)

    def on_throttle(self):
        # Back off aggressively
        self.current_delay = min(self.max_delay, self.current_delay * self.backoff_factor)

    def get_delay(self) -> float:
        # Add jitter around current_delay
        return random.uniform(self.current_delay, self.current_delay * 1.5)
```

Replace the fixed `_random_delay()` in `QanoonScraper` with a call to this controller when `ADAPTIVE_THROTTLING=true`.

### 3. Dynamic Worker Pool

Extend the `ThreadPoolExecutor` logic to resize based on recent error rate:

- Track a rolling window error rate (e.g., last 50 requests).
- If error rate exceeds 10%, reduce `max_workers` by half (minimum 1).
- If error rate stays below 2% for 60 seconds, increase `max_workers` by 1 (up to configured maximum).

This avoids the need to restart the process to react to throttling.

### 4. Identity Rotation (No Proxy)

Even without a proxy pool, rotate the following on throttle signals:

- `User-Agent` from a curated list of desktop browsers.
- `Accept-Language` ordering.
- Optional: TLS fingerprint via different `requests` adapters or switching to Playwright.

### 5. CAPTCHA Path

If a CAPTCHA is detected:

1. Pause scraping for that worker.
2. Log the URL and HTML snippet.
3. Option A: Integrate a CAPTCHA-solving service (2Captcha, Anti-Captcha) via an adapter.
4. Option B: Move the URL to a "human queue" and continue with other documents.
5. After solving, resume with increased delays to avoid re-triggering.

### 6. Configuration

Add these environment variables to `.env`:

```bash
ADAPTIVE_THROTTLING=true
ADAPTIVE_MIN_DELAY=0.2
ADAPTIVE_MAX_DELAY=30.0
ADAPTIVE_BACKOFF_FACTOR=2.0
ADAPTIVE_ERROR_THRESHOLD=0.10
ROTATE_USER_AGENT=true
CAPTCHA_SERVICE=none  # options: none, 2captcha, anticaptcha
```

### 7. Integration Points (No Code Changes Today)

To implement this later without disrupting the current scraper:

- Wrap `fetch_page()` in a decorator that records response signals.
- Add the `AdaptiveDelayController` as an optional constructor argument in `QanoonScraper`.
- Use a worker-pool manager instead of a fixed `ThreadPoolExecutor(max_workers)`.
- Keep the existing fixed-delay path as the default (`ADAPTIVE_THROTTLING=false`).

### 8. Expected Impact

- **Before**: fixed ~0.2–0.7 s delays, fixed worker count.
- **After**: delays and workers self-tune based on site behavior, reducing hard blocks and improving average throughput under variable conditions.

## Conclusion

This plan keeps the current scraper intact while providing a clear, configurable path to handle dynamic throttling and CAPTCHA challenges. The modular fetcher design already supports most of these changes with minimal refactoring.
