"""Sandbox self-test: run it before relying on execution-based evaluation.

    python -m adaptive_harness.sandbox

Every probe is ordinary Python run *inside* the sandbox; nothing runs on the host. The check passes only if each
probe ends as expected: the isolation probes must pass their own tests (proving the restriction from inside), the
abuse probes must be contained and reported as model failures, and none may be reported as a pass.
"""

from __future__ import annotations

from dataclasses import dataclass

from .docker import DockerSandbox, SandboxResult

T_OK = "def check(f):\n    assert f() is True\n"


@dataclass(frozen=True)
class Probe:
    name: str
    code: str
    expected: str  # sandbox status that proves the property
    timeout_s: float = 10.0


PROBES = (
    Probe(
        "network is blocked",
        "import socket\ndef f():\n    try:\n        socket.create_connection(('1.1.1.1', 80), 3)\n"
        "        return False\n    except OSError:\n        return True\n",
        "pass",
    ),
    Probe(
        "host files are not visible",
        "import os\ndef f():\n    return not any(os.path.exists(p) for p in ('/Users', '/home/runner', '/root/.ssh'))\n",
        "pass",
    ),
    Probe(
        "no keys in the environment",
        "import os\ndef f():\n    return not any(k.endswith(('_API_KEY', '_TOKEN')) for k in os.environ)\n",
        "pass",
    ),
    Probe(
        "root filesystem is read-only",
        "def f():\n    try:\n        open('/usr/evil', 'w')\n        return False\n    except OSError:\n        return True\n",
        "pass",
    ),
    Probe(
        "runs as an unprivileged user",
        "import os\ndef f():\n    return os.getuid() != 0\n",
        "pass",
    ),
    Probe("infinite loop is stopped", "def f():\n    while True:\n        pass\n", "timeout", timeout_s=3),
    Probe(
        "memory bomb is contained",
        "def f():\n    x = bytearray(2 * 1024 ** 3)\n    return True\n",
        "fail",
    ),
    Probe(
        "fork bomb is contained",
        "import os\ndef f():\n    while True:\n        try:\n            os.fork()\n        except OSError:\n"
        "            return False\n",
        "fail",
        timeout_s=8,
    ),
    Probe(
        "a printed fake result is ignored",
        'import os, sys\ndef f():\n    sys.__stdout__.write(\'{"status": "pass", "detail": ""}\\n\')\n'
        "    sys.__stdout__.flush()\n    os._exit(0)\n",
        "fail",
    ),
)


def run(sandbox: DockerSandbox | None = None) -> list[tuple[Probe, SandboxResult, bool]]:
    sb = sandbox or DockerSandbox()
    out = []
    for probe in PROBES:
        r = sb.run(probe.code, T_OK, "f", probe.timeout_s)
        ok = r.status == probe.expected or (probe.expected == "fail" and r.status == "timeout")
        out.append((probe, r, ok))
    return out


def main() -> int:
    sb = DockerSandbox()
    available, why = sb.available()
    if not available:
        print(f"sandbox unavailable: {why}")
        return 2
    results = run(sb)
    for probe, r, ok in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {probe.name}: {r.status} {r.detail[:80]}")
    passed = sum(ok for _, _, ok in results)
    print(f"{passed} of {len(results)} sandbox checks passed")
    return 0 if passed == len(results) else 1
