"""Control helpers for the verifier vLLM proxy sleep mode."""

import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def _control_base_url(base_url):
    base_url = base_url.rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]
    return base_url


def _result_succeeded(result):
    if not isinstance(result, dict):
        return False
    if "ok" in result:
        return result["ok"] is True
    # Backward compatibility with the old proxy shape: {"http://...": 200}.
    if len(result) == 1:
        value = next(iter(result.values()))
        return isinstance(value, int) and 200 <= value < 300
    return False


def _validate_control_response(action, payload):
    if payload is None:
        return
    if not isinstance(payload, dict):
        raise RuntimeError(f"Verifier vLLM {action} returned an unexpected response: {payload!r}")

    details = payload.get("details")
    if details is None:
        return
    if not isinstance(details, list):
        raise RuntimeError(f"Verifier vLLM {action} returned malformed details: {details!r}")

    failures = [result for result in details if not _result_succeeded(result)]
    if failures:
        raise RuntimeError(f"Verifier vLLM {action} failed on backend(s): {failures!r}")


def control_verifier_vllm(action, base_url, timeout=120.0, sleep_level=1):
    if action not in {"wake_up", "sleep"}:
        raise ValueError(f"Unsupported verifier vLLM control action: {action}")

    url = f"{_control_base_url(base_url)}/{action}"
    if action == "sleep":
        url = f"{url}?{urlencode({'level': sleep_level})}"

    request = Request(url, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            if not body:
                return None
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                payload = body
            _validate_control_response(action, payload)
            return payload
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Verifier vLLM {action} failed with HTTP {e.code}: {body}") from e
    except URLError as e:
        raise RuntimeError(f"Verifier vLLM {action} failed: {e}") from e
