"""Host side of the Docker sandbox (DESIGN §5.7).

Each job runs in a fresh container with no network, a read-only root filesystem, tmpfs scratch space, an
unprivileged user, no capabilities, and limits on processes, memory and CPU. No host directory is mounted and no
host environment variable is passed. If Docker or the image is unavailable, the result is ``infra_error``;
nothing ever falls back to running code on the host.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from typing import Literal

from .runner import BOOTSTRAP, CHILD, RUNNER

# python:3.12-slim, pinned by digest (multi-architecture index), recorded 2026-11-10.
DEFAULT_IMAGE = "python:3.12-slim@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159"

Status = Literal["pass", "fail", "timeout", "env_error", "infra_error"]


@dataclass(frozen=True)
class SandboxResult:
    status: Status
    detail: str
    seconds: float | None = None


class DockerSandbox:
    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        *,
        docker: str = "docker",
        memory: str = "512m",
        cpus: str = "1",
        pids: int = 128,
        startup_allowance_s: float = 60.0,
    ) -> None:
        self.image = image
        self.docker = docker
        self.memory = memory
        self.cpus = cpus
        self.pids = pids
        self.startup_allowance_s = startup_allowance_s

    def command(self, name: str) -> list[str]:
        return [
            self.docker, "run", "--rm", "-i", "--name", name,
            "--pull", "never",
            "--network", "none",
            "--read-only",
            "--tmpfs", "/work:rw,size=64m,mode=1777",
            "--tmpfs", "/tmp:rw,size=64m,mode=1777",
            "--user", "65534:65534",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--pids-limit", str(self.pids),
            "--memory", self.memory,
            "--memory-swap", self.memory,
            "--cpus", self.cpus,
            "--workdir", "/work",
            self.image,
            "python", "-I", "-c", BOOTSTRAP,
        ]  # fmt: skip

    def available(self) -> tuple[bool, str]:
        if shutil.which(self.docker) is None:
            return False, f"{self.docker} not found"
        try:
            p = subprocess.run(
                [self.docker, "image", "inspect", self.image], capture_output=True, text=True, timeout=30
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            return False, f"docker not usable: {e}"
        if p.returncode != 0:
            return (
                False,
                f"image not available locally (run: docker pull {self.image}): {p.stderr.strip()[:200]}",
            )
        return True, "ok"

    def run(self, code: str, test: str, entry: str, timeout_s: float = 10.0) -> SandboxResult:
        """Run ``test``'s ``check(candidate)`` against ``entry`` from ``code``."""
        name = f"adaptive-harness-{uuid.uuid4().hex[:12]}"
        payload = json.dumps(
            {
                "runner": RUNNER,
                "child": CHILD,
                "code": code,
                "test": test,
                "entry": entry,
                "timeout": timeout_s,
            }
        )
        t0 = time.time()
        try:
            p = subprocess.run(
                self.command(name),
                input=payload,
                capture_output=True,
                text=True,
                timeout=timeout_s + self.startup_allowance_s,
            )
        except FileNotFoundError as e:
            return SandboxResult("infra_error", f"docker not found: {e}")
        except subprocess.TimeoutExpired:
            subprocess.run([self.docker, "rm", "-f", name], capture_output=True, timeout=30)
            return SandboxResult(
                "infra_error", "container did not finish in time", round(time.time() - t0, 3)
            )
        lines = [ln for ln in p.stdout.strip().splitlines() if ln.startswith("{")]
        if not lines:
            return SandboxResult(
                "infra_error",
                f"no result from container (exit {p.returncode}): {p.stderr.strip()[-300:]}",
                round(time.time() - t0, 3),
            )
        try:
            r = json.loads(lines[-1])
        except ValueError:
            return SandboxResult("infra_error", "unparseable result from container")
        status = r.get("status")
        if status not in ("pass", "fail", "timeout", "env_error"):
            return SandboxResult("infra_error", f"unexpected status {status!r}")
        return SandboxResult(status, str(r.get("detail", "")), r.get("seconds"))
