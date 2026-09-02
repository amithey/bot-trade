"""
Tests for docker/entrypoint.sh — the thing that turns a Fly/Railway/Render
secret into `.streamlit/secrets.toml` before Streamlit starts.

docker-compose bind-mounts the file directly (its own commented-out volume
line in docker-compose.yml), so this script only matters on hosts with no way
to mount a host file in — which is exactly the case this project's own Fly
deployment hit. These tests run the real shell script (not a reimplementation
of its logic) via bash, pointed at a scratch directory instead of the real
/app via BOTTRADE_APP_DIR.
"""
from __future__ import annotations

import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "docker" / "entrypoint.sh"


def _find_bash() -> str:
    """A real bash — on Windows, `shutil.which` alone is not reliable.

    This machine has two things named bash.exe on PATH: Git for Windows'
    real one, and the legacy `C:\\Windows\\System32\\bash.exe`, which is a
    shim for the WSL feature and fails outright
    (`execvpe(/bin/bash) failed: No such file or directory`) unless a WSL
    distro is actually installed. Which one plain `"bash"` resolves to
    depends on PATH order and isn't consistent between this process and the
    one launching it, so prefer Git's bash by known install path first.
    """
    for candidate in (r"C:\Program Files\Git\bin\bash.exe",
                      r"C:\Program Files\Git\usr\bin\bash.exe"):
        if Path(candidate).exists():
            return candidate
    found = shutil.which("bash")
    if found:
        return found
    pytest.skip("no usable bash found")


_BASH = _find_bash() if sys.platform == "win32" else "bash"


def _run(env: dict, args: list[str] = None) -> subprocess.CompletedProcess:
    # A relative script path with cwd=_ROOT, not str(_SCRIPT): on Windows
    # git-bash, handing bash an absolute `C:\...\entrypoint.sh` string
    # verbatim doesn't resolve — bash only understands its own
    # MSYS-translated form (`/c/...`), and the repo's own path here has
    # non-ASCII directory components on top of that. Running from the repo
    # root and letting bash resolve the relative path through its already-
    # translated CWD is what actually works cross-platform; explicit utf-8
    # with errors replaced keeps any incidental non-ASCII output from
    # crashing the capture on Windows' default console encoding.
    return subprocess.run(
        [_BASH, "docker/entrypoint.sh", *(args or ["true"])],
        cwd=_ROOT, env=env, capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    )


def _base_env(app_dir: Path) -> dict:
    import os
    return {**os.environ, "BOTTRADE_APP_DIR": str(app_dir)}


def test_script_exists_and_is_referenced_by_the_dockerfile():
    assert _SCRIPT.exists()
    dockerfile = (_SCRIPT.parent.parent / "Dockerfile").read_text(encoding="utf-8")
    assert "docker/entrypoint.sh" in dockerfile
    assert "chmod +x" in dockerfile


def test_no_op_when_the_secret_is_unset(tmp_path):
    env = _base_env(tmp_path)
    env.pop("BOTTRADE_STREAMLIT_SECRETS", None)
    result = _run(env)
    assert result.returncode == 0
    assert not (tmp_path / ".streamlit").exists()


def test_no_op_when_the_secret_is_set_but_empty(tmp_path):
    """An empty string is not meaningfully "set" for this purpose — writing
    an empty secrets.toml would just make OIDC fail differently."""
    env = _base_env(tmp_path)
    env["BOTTRADE_STREAMLIT_SECRETS"] = ""
    result = _run(env)
    assert result.returncode == 0
    assert not (tmp_path / ".streamlit").exists()


def test_writes_the_exact_secret_content(tmp_path):
    content = '[auth]\nclient_id = "abc"\ncookie_secret = "xyz"\n'
    env = _base_env(tmp_path)
    env["BOTTRADE_STREAMLIT_SECRETS"] = content
    result = _run(env)
    assert result.returncode == 0
    written = tmp_path / ".streamlit" / "secrets.toml"
    assert written.exists()
    assert written.read_text(encoding="utf-8") == content


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="NTFS has no real POSIX permission bits, and git-bash's chmod/"
           "install -m only approximate them — the script's real target, the "
           "Linux container it ships in, honours 600 exactly (verified "
           "separately in CI, which runs this same test on ubuntu-latest).",
)
def test_secrets_file_is_not_world_or_group_readable(tmp_path):
    env = _base_env(tmp_path)
    env["BOTTRADE_STREAMLIT_SECRETS"] = "[auth]\nclient_secret = \"real-secret\"\n"
    _run(env)
    written = tmp_path / ".streamlit" / "secrets.toml"
    mode = stat.S_IMODE(written.stat().st_mode)
    assert mode == 0o600, oct(mode)


def test_execs_the_given_command_and_forwards_its_exit_code(tmp_path):
    env = _base_env(tmp_path)
    ok = _run(env, args=["true"])
    assert ok.returncode == 0
    bad = _run(env, args=["false"])
    assert bad.returncode != 0


def test_forwards_stdout_from_the_wrapped_command(tmp_path):
    env = _base_env(tmp_path)
    result = _run(env, args=["echo", "hello-from-streamlit"])
    assert "hello-from-streamlit" in result.stdout


def test_overwrites_a_stale_secrets_files_content_from_a_previous_boot(tmp_path):
    """A redeploy with a rotated secret must not leave the old file's
    content behind."""
    streamlit_dir = tmp_path / ".streamlit"
    streamlit_dir.mkdir()
    stale = streamlit_dir / "secrets.toml"
    stale.write_text("[auth]\nclient_secret = \"old-and-rotated\"\n", encoding="utf-8")

    env = _base_env(tmp_path)
    env["BOTTRADE_STREAMLIT_SECRETS"] = "[auth]\nclient_secret = \"new\"\n"
    _run(env)

    assert stale.read_text(encoding="utf-8") == '[auth]\nclient_secret = "new"\n'


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="NTFS has no real POSIX permission bits, and git-bash's chmod/"
           "install -m only approximate them — the script's real target, the "
           "Linux container it ships in, honours 600 exactly (verified "
           "separately in CI, which runs this same test on ubuntu-latest).",
)
def test_overwriting_a_stale_secrets_file_still_tightens_its_permissions(tmp_path):
    """A file left world-readable by a previous, less careful boot must not
    stay that way just because the file already existed."""
    streamlit_dir = tmp_path / ".streamlit"
    streamlit_dir.mkdir()
    stale = streamlit_dir / "secrets.toml"
    stale.write_text("[auth]\nclient_secret = \"old\"\n", encoding="utf-8")
    stale.chmod(0o644)

    env = _base_env(tmp_path)
    env["BOTTRADE_STREAMLIT_SECRETS"] = "[auth]\nclient_secret = \"new\"\n"
    _run(env)

    assert stat.S_IMODE(stale.stat().st_mode) == 0o600
