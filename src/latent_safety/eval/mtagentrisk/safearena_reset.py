from __future__ import annotations

import argparse
import json
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener


EXPECTED_CONTAINERS = {
    "mtagentrisk_sa_forum",
    "mtagentrisk_sa_shopping",
    "mtagentrisk_sa_shopping_admin",
}


class SafeArenaResetError(RuntimeError):
    pass


def _request_json(
    url: str,
    *,
    method: str = "GET",
    timeout_seconds: float,
) -> dict[str, Any]:
    request = Request(
        url,
        method=method,
        headers={"Accept": "application/json"},
    )
    opener = build_opener(ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout_seconds) as response:
            payload = response.read().decode("utf-8")
    except (HTTPError, URLError, TimeoutError) as exc:
        raise SafeArenaResetError(f"SafeArena reset request failed: {url}: {exc}") from exc
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise SafeArenaResetError(
            f"SafeArena reset returned non-JSON data: {url}: {payload[:200]!r}"
        ) from exc
    if not isinstance(decoded, dict):
        raise SafeArenaResetError(f"SafeArena reset returned non-object JSON: {url}")
    return decoded


def _result_container_ids(result: object) -> dict[str, str]:
    if not isinstance(result, dict):
        return {}
    rows = result.get("containers")
    if not isinstance(rows, list):
        return {}
    identifiers: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = row.get("name")
        identifier = row.get("id")
        if isinstance(name, str) and isinstance(identifier, str):
            identifiers[name] = identifier
    return identifiers


def validate_completed_health(
    health: dict[str, Any],
    *,
    previous_ids: dict[str, str] | None = None,
) -> dict[str, Any]:
    if health.get("ok") is not True or health.get("status") != "ok":
        raise SafeArenaResetError(f"SafeArena bridge is unhealthy: {health!r}")
    if health.get("busy") is not False:
        raise SafeArenaResetError(f"SafeArena reset is still busy: {health!r}")
    result = health.get("last_result")
    if not isinstance(result, dict):
        raise SafeArenaResetError("SafeArena bridge has no completed reset result")
    if (
        result.get("ok") is not True
        or result.get("exit_code") != 0
        or result.get("status") != "success"
    ):
        raise SafeArenaResetError(f"SafeArena reset failed: {result!r}")

    rows = result.get("containers")
    if not isinstance(rows, list):
        raise SafeArenaResetError("SafeArena reset result has no container rows")
    statuses = {
        row.get("name"): row.get("status")
        for row in rows
        if isinstance(row, dict)
    }
    if set(statuses) != EXPECTED_CONTAINERS:
        raise SafeArenaResetError(
            f"SafeArena reset returned unexpected containers: {sorted(statuses)}"
        )
    not_running = sorted(name for name, status in statuses.items() if status != "running")
    if not_running:
        raise SafeArenaResetError(
            f"SafeArena reset containers are not running: {not_running}"
        )

    identifiers = _result_container_ids(result)
    if set(identifiers) != EXPECTED_CONTAINERS:
        raise SafeArenaResetError("SafeArena reset result is missing container IDs")
    if previous_ids is not None:
        unchanged = sorted(
            name
            for name in EXPECTED_CONTAINERS
            if previous_ids.get(name) == identifiers.get(name)
        )
        if unchanged:
            raise SafeArenaResetError(
                f"SafeArena reset did not recreate containers: {unchanged}"
            )
    return result


def _wait_until_idle(
    base_url: str,
    *,
    deadline: float,
    poll_interval_seconds: float,
) -> dict[str, Any]:
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise SafeArenaResetError("Timed out waiting for SafeArena reset bridge")
        try:
            health = _request_json(
                f"{base_url}/health",
                timeout_seconds=min(10.0, remaining),
            )
        except SafeArenaResetError:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SafeArenaResetError(
                    "Timed out waiting for SafeArena reset bridge"
                )
            time.sleep(min(poll_interval_seconds, remaining))
            continue
        if health.get("ok") is not True or health.get("status") != "ok":
            raise SafeArenaResetError(f"SafeArena bridge is unhealthy: {health!r}")
        if health.get("busy") is False:
            return health
        time.sleep(min(poll_interval_seconds, max(0.0, remaining)))


def perform_reset(
    base_url: str,
    *,
    timeout_seconds: float = 300.0,
    poll_interval_seconds: float = 2.0,
) -> dict[str, Any]:
    base_url = base_url.rstrip("/")
    deadline = time.monotonic() + timeout_seconds
    before = _wait_until_idle(
        base_url,
        deadline=deadline,
        poll_interval_seconds=poll_interval_seconds,
    )
    previous_ids = _result_container_ids(before.get("last_result")) or None

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise SafeArenaResetError("Timed out before starting SafeArena reset")
    _request_json(
        f"{base_url}/reset",
        method="POST",
        timeout_seconds=remaining,
    )
    after = _wait_until_idle(
        base_url,
        deadline=deadline,
        poll_interval_seconds=poll_interval_seconds,
    )
    return validate_completed_health(after, previous_ids=previous_ids)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reset tunneled SafeArena services and validate container recreation."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:21009")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=2.0)
    args = parser.parse_args()

    result = perform_reset(
        args.base_url,
        timeout_seconds=args.timeout_seconds,
        poll_interval_seconds=args.poll_interval_seconds,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "duration_seconds": result.get("duration_seconds"),
                "ended_at": result.get("ended_at"),
                "container_ids": _result_container_ids(result),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
