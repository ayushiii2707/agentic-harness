"""Every code calibration task must be solvable: a reference solution passes its test in the real sandbox,
and an obviously wrong solution fails it."""

from __future__ import annotations

import pytest

from adaptive_harness.calibration.suites import code_suite
from adaptive_harness.evaluation import EvalStatus
from adaptive_harness.sandbox import DockerSandbox

SANDBOX = DockerSandbox()
AVAILABLE, WHY = SANDBOX.available()
pytestmark = [pytest.mark.sandbox, pytest.mark.skipif(not AVAILABLE, reason=f"sandbox unavailable: {WHY}")]

REFERENCES = {
    "is_prime": "def is_prime(n):\n    if n < 2: return False\n    i = 2\n    while i * i <= n:\n        if n % i == 0: return False\n        i += 1\n    return True",
    "reverse_words": "def reverse_words(s):\n    return ' '.join(reversed(s.split()))",
    "fizzbuzz": "def fizzbuzz(n):\n    return ['FizzBuzz' if i % 15 == 0 else 'Fizz' if i % 3 == 0 else 'Buzz' if i % 5 == 0 else str(i) for i in range(1, n + 1)]",
    "flatten": "def flatten(xs):\n    out = []\n    for x in xs:\n        out.extend(flatten(x) if isinstance(x, list) else [x])\n    return out",
    "is_palindrome": "def is_palindrome(s):\n    t = [c.lower() for c in s if c.isalnum()]\n    return t == t[::-1]",
    "merge_sorted": "def merge_sorted(a, b):\n    i = j = 0\n    out = []\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]: out.append(a[i]); i += 1\n        else: out.append(b[j]); j += 1\n    return out + a[i:] + b[j:]",
    "roman_to_int": "def roman_to_int(s):\n    v = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}\n    t = 0\n    for i, c in enumerate(s):\n        t += -v[c] if i + 1 < len(s) and v[c] < v[s[i+1]] else v[c]\n    return t",
    "int_to_roman": "def int_to_roman(n):\n    p = [(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),(90,'XC'),(50,'L'),(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]\n    out = ''\n    for v, s in p:\n        while n >= v: out += s; n -= v\n    return out",
    "anagrams": "def anagrams(a, b):\n    k = lambda s: sorted(s.replace(' ', '').lower())\n    return k(a) == k(b)",
    "gcd": "def gcd(a, b):\n    while b: a, b = b, a % b\n    return a",
    "run_length": "def run_length(s):\n    out, i = '', 0\n    while i < len(s):\n        j = i\n        while j < len(s) and s[j] == s[i]: j += 1\n        out += s[i] + str(j - i); i = j\n    return out",
    "chunk": "def chunk(xs, n):\n    return [xs[i:i + n] for i in range(0, len(xs), n)]",
    "second_largest": "def second_largest(xs):\n    u = sorted(set(xs))\n    return u[-2] if len(u) >= 2 else None",
    "word_count": "import re\nfrom collections import Counter\ndef word_count(s):\n    return dict(Counter(re.findall(r'[a-z]+', s.lower())))",
    "binary_search": "def binary_search(xs, x):\n    lo, hi = 0, len(xs) - 1\n    while lo <= hi:\n        m = (lo + hi) // 2\n        if xs[m] == x: return m\n        if xs[m] < x: lo = m + 1\n        else: hi = m - 1\n    return -1",
    "transpose": "def transpose(m):\n    return [list(r) for r in zip(*m)]",
    "caesar": "def caesar(s, k):\n    out = ''\n    for c in s:\n        if c.isalpha():\n            b = ord('A') if c.isupper() else ord('a')\n            out += chr((ord(c) - b + k) % 26 + b)\n        else: out += c\n    return out",
    "valid_parens": "def valid_parens(s):\n    st, pairs = [], {')':'(', ']':'[', '}':'{'}\n    for c in s:\n        if c in '([{': st.append(c)\n        elif c in pairs:\n            if not st or st.pop() != pairs[c]: return False\n    return not st",
    "digits_sum": "def digits_sum(n):\n    while n >= 10: n = sum(int(d) for d in str(n))\n    return n",
    "dedupe": "def dedupe(xs):\n    seen, out = set(), []\n    for x in xs:\n        if x not in seen: seen.add(x); out.append(x)\n    return out",
    "rotate": "def rotate(xs, k):\n    if not xs: return []\n    k %= len(xs)\n    return xs[-k:] + xs[:-k] if k else list(xs)",
    "median": "def median(xs):\n    s = sorted(xs); n = len(s)\n    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2",
    "title_case": "def title_case(s):\n    return ' '.join(w[:1].upper() + w[1:].lower() for w in s.split(' '))",
    "pascal_row": "def pascal_row(n):\n    row = [1]\n    for _ in range(n): row = [a + b for a, b in zip([0] + row, row + [0])]\n    return row",
    "max_subarray": "def max_subarray(xs):\n    best = cur = xs[0]\n    for x in xs[1:]:\n        cur = max(x, cur + x); best = max(best, cur)\n    return best",
    "to_snake": "import re\ndef to_snake(s):\n    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\\1_\\2', s)\n    return re.sub(r'([a-z0-9])([A-Z])', r'\\1_\\2', s).lower()",
    "count_islands": "def count_islands(g):\n    if not g: return 0\n    seen, n = set(), 0\n    for i in range(len(g)):\n        for j in range(len(g[0])):\n            if g[i][j] == 1 and (i, j) not in seen:\n                n += 1; st = [(i, j)]\n                while st:\n                    a, b = st.pop()\n                    if (a, b) in seen or not (0 <= a < len(g) and 0 <= b < len(g[0])) or g[a][b] != 1: continue\n                    seen.add((a, b)); st += [(a+1,b),(a-1,b),(a,b+1),(a,b-1)]\n    return n",
    "lcs_length": "def lcs_length(a, b):\n    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]\n    for i in range(len(a)):\n        for j in range(len(b)):\n            d[i+1][j+1] = d[i][j] + 1 if a[i] == b[j] else max(d[i][j+1], d[i+1][j])\n    return d[-1][-1]",
    "primes_upto": "def primes_upto(n):\n    if n < 2: return []\n    s = [True] * (n + 1); s[0] = s[1] = False\n    for i in range(2, int(n ** 0.5) + 1):\n        if s[i]: s[i*i::i] = [False] * len(s[i*i::i])\n    return [i for i, v in enumerate(s) if v]",
    "parse_duration": "import re\ndef parse_duration(s):\n    u = {'h': 3600, 'm': 60, 's': 1}\n    return sum(int(n) * u[k] for n, k in re.findall(r'(\\d+)([hms])', s))",
}

TASKS = code_suite()


def test_every_task_has_a_reference() -> None:
    assert {t.evaluator.entry for t in TASKS} == set(REFERENCES)  # type: ignore[attr-defined]


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.evaluator.entry)  # type: ignore[attr-defined]
def test_reference_passes_and_wrong_answer_fails(task) -> None:  # type: ignore[no-untyped-def]
    entry = task.evaluator.entry
    good = task.evaluator.evaluate(f"```python\n{REFERENCES[entry]}\n```")
    assert good.status is EvalStatus.SUCCESS, good.details
    bad = task.evaluator.evaluate(f"```python\ndef {entry}(*args, **kwargs):\n    return None\n```")
    assert bad.status is EvalStatus.FAILURE, bad.details
