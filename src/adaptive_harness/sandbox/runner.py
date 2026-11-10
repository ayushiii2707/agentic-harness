"""Code that runs *inside* the container (DESIGN §5.7).

RUNNER is trusted: it receives the job on stdin (via BOOTSTRAP), writes the files to the container's tmpfs, starts CHILD as a
separate process, and prints exactly one JSON result line on the container's stdout.

CHILD runs the untrusted solution and the test. Its stdout and stderr go to pipes read by RUNNER, never to the
container's stdout, so the solution cannot print a fake result. CHILD reports through a private pipe, and the
report must carry a per-job random nonce that is read and closed before the solution is imported. A solution
that exits early or crashes produces no valid report, which counts as a failure.

Residual risk, stated honestly: the solution and the test share the CHILD interpreter, so a deliberately
adversarial solution could still introspect the test harness. The threat model is non-adversarial model output;
the container boundary is what protects the host.
"""

BOOTSTRAP = "import json,sys; d=sys.stdin.read(); exec(json.loads(d)['runner'])"
"""Command-line program run by the container's python: read the job, then execute RUNNER with ``d`` in scope."""

RUNNER = r"""
import json, os, secrets, subprocess, sys, time
job = json.loads(d)
os.makedirs("/work/job", exist_ok=True)
with open("/work/job/solution.py", "w", encoding="utf-8") as f:
    f.write(job["code"])
with open("/work/job/spec.json", "w", encoding="utf-8") as f:
    json.dump({"test": job["test"], "entry": job["entry"]}, f)
with open("/work/job/_child.py", "w", encoding="utf-8") as f:
    f.write(job["child"])
nonce = secrets.token_hex(16)
nr, nw = os.pipe()
rr, rw = os.pipe()
os.write(nw, nonce.encode())
os.close(nw)
env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/work", "PYTHONDONTWRITEBYTECODE": "1", "LANG": "C.UTF-8"}
t0 = time.time()
p = subprocess.Popen([sys.executable, "-I", "/work/job/_child.py", str(nr), str(rw)], pass_fds=(nr, rw),
                     stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd="/work/job", env=env)
os.close(nr)
os.close(rw)
status, detail = None, ""
try:
    _, err = p.communicate(timeout=job["timeout"])
except subprocess.TimeoutExpired:
    p.kill()
    _, err = p.communicate()
    status, detail = "timeout", f"exceeded {job['timeout']} s"
raw = b""
while True:
    chunk = os.read(rr, 65536)
    if not chunk:
        break
    raw += chunk
if status is None:
    try:
        rep = json.loads(raw.decode("utf-8", "replace") or "{}")
    except ValueError:
        rep = {}
    if rep.get("nonce") == nonce and rep.get("status") in ("pass", "fail", "env_error"):
        status, detail = rep["status"], rep.get("detail", "")
    else:
        tail = err.decode("utf-8", "replace").strip().splitlines()[-2:]
        status = "fail"
        detail = f"no verified result (exit code {p.returncode}) " + " | ".join(tail)
print(json.dumps({"status": status, "detail": detail[:600], "seconds": round(time.time() - t0, 3)}), flush=True)
"""

CHILD = r"""
import json, os, sys


def main():
    nr, rw = int(sys.argv[1]), int(sys.argv[2])
    token = os.read(nr, 64).decode()
    os.close(nr)
    out = os.fdopen(rw, "w")

    def emit(status, detail=""):
        out.write(json.dumps({"nonce": token, "status": status, "detail": str(detail)[:600]}))
        out.flush()

    spec = json.load(open("/work/job/spec.json", encoding="utf-8"))
    sys.path.insert(0, "/work/job")
    try:
        import solution
    except ModuleNotFoundError as e:
        emit("env_error", f"missing module: {e.name}")
        return
    except BaseException as e:
        emit("fail", f"import failed: {type(e).__name__}: {e}")
        return
    fn = getattr(solution, spec["entry"], None)
    if fn is None:
        emit("fail", f"entry point {spec['entry']!r} not defined")
        return
    ns = {"__name__": "harness_test"}
    try:
        exec(compile(spec["test"], "test.py", "exec"), ns)
        ns["check"](fn)
    except AssertionError as e:
        emit("fail", f"AssertionError: {e}")
        return
    except BaseException as e:
        emit("fail", f"{type(e).__name__}: {e}")
        return
    emit("pass")


main()
"""
