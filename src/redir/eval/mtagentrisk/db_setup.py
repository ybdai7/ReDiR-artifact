from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


_CREATED_ROLE_PATTERN = re.compile(
    r"(?im)^\s*CREATE\s+(?:ROLE|USER)\s+([A-Za-z_][A-Za-z0-9_$]*)\s*;"
)


def _postgres_psql_command(*args: str) -> list[str]:
    container = os.environ.get("MCP_POSTGRES_CONTAINER", "mcpmark-postgres")
    shell_command = (
        'exec env PGPASSWORD="$POSTGRES_PASSWORD" '
        'psql -h 127.0.0.1 '
        '-U "${POSTGRES_USER:-postgres}" '
        '-d "${POSTGRES_DB:-postgres}" '
        '-v ON_ERROR_STOP=1 "$@"'
    )
    return [
        "docker",
        "exec",
        "-i",
        container,
        "sh",
        "-c",
        shell_command,
        "mtagentrisk-psql",
        *args,
    ]


def probe_postgres_db() -> bool:
    """Check the exact authenticated database path used by task resets."""
    try:
        result = subprocess.run(
            _postgres_psql_command("-Atqc", "SELECT 1"),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"[ERROR] PostgreSQL authenticated probe failed: {exc}")
        return False
    return result.stdout.strip() == "1"


def _roles_created_by_seed(seed_sql: str) -> list[str]:
    return list(dict.fromkeys(_CREATED_ROLE_PATTERN.findall(seed_sql)))


def _drop_seed_roles(role_names: list[str]) -> None:
    if not role_names:
        return
    array = ", ".join(f"'{name}'" for name in role_names)
    cleanup_sql = f"""
DO $mtagentrisk$
DECLARE role_name text;
BEGIN
  FOREACH role_name IN ARRAY ARRAY[{array}]
  LOOP
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = role_name) THEN
      EXECUTE format('DROP OWNED BY %I', role_name);
      EXECUTE format('DROP ROLE %I', role_name);
    END IF;
  END LOOP;
END
$mtagentrisk$;
"""
    subprocess.run(
        _postgres_psql_command("-Atqc", cleanup_sql),
        check=True,
        stdout=subprocess.DEVNULL,
    )


def reset_postgres_db(seed_file_path: str) -> bool:
    """Reset PostgreSQL from a task seed through the container's own credentials."""
    seed_path = Path(seed_file_path).expanduser().resolve()
    if not seed_path.is_file():
        print(f"[ERROR] Seed file not found at: {seed_path}")
        return False

    try:
        seed_sql = seed_path.read_text(encoding="utf-8")
        _drop_seed_roles(_roles_created_by_seed(seed_sql))
        with seed_path.open("rb") as seed_file:
            subprocess.run(
                _postgres_psql_command(),
                stdin=seed_file,
                check=True,
                stdout=subprocess.DEVNULL,
            )
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"[ERROR] Failed to reset PostgreSQL database: {exc}")
        return False

    print("[INFO] PostgreSQL database reset successfully.")
    return True
