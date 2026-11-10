"""Calibration suites (DESIGN §6.5, §13): our own tasks, with deterministic primary evaluators.

- ``math``: generated multi-step word problems with computed answers (ReferenceAnswer).
- ``structured_output``: generated extraction tasks with exact expected field values (JsonFields).
- ``code_generation``: hand-written function tasks with unit tests run in the Docker sandbox (UnitTests).
- ``instructions``: generated instruction-following tasks with several exact constraints (BulletConstraints).
  Harder than the other suites on purpose: strong models reach ceiling on math and extraction, and a suite at
  ceiling cannot tell strategies apart.

Generated suites take a ``split``: ``calibration`` (used to learn) and ``heldout`` (reserved for Phase 12) come
from disjoint seeds, so the evaluation of a learned profile never reuses the tasks it was learned on.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

from ..evaluation import Evaluator
from ..evaluation.builtin import BulletConstraints, JsonFields, ReferenceAnswer, UnitTests

Split = Literal["calibration", "heldout"]
SUITE_VERSION = "1"
_SEEDS = {"calibration": 1001, "heldout": 2002}


@dataclass(frozen=True)
class CalibrationTask:
    id: str
    category: str
    prompt: str
    evaluator: Evaluator


# ------------------------------------------------------------------ math
def _math_problem(rng: random.Random) -> tuple[str, float]:
    kind = rng.randrange(6)
    if kind == 0:
        k, price, packs = rng.randint(2, 6), rng.randint(2, 9), rng.randint(3, 12)
        n = k * packs
        disc = rng.choice([10, 20, 25])
        return (
            f"A shop sells notebooks at {k} for ${price}. Maya buys {n} notebooks and gets {disc}% off the total. "
            f"How many dollars does she pay?",
            packs * price * (100 - disc) / 100,
        )
    if kind == 1:
        a = rng.randint(3, 9)
        b = rng.choice([x for x in range(2, 13) if x != a])
        d = b * rng.randint(1, 3)  # makes a * d / b a whole number
        extra = rng.randint(1, 5)
        return (
            f"{a} workers finish a job in {d} days. How many days would {b} workers take at the same rate, "
            f"plus {extra} extra days for inspection?",
            a * d / b + extra,
        )
    if kind == 2:
        c = rng.choice([120, 180, 240, 360, 600])
        r1 = rng.randint(8, 20)
        r2 = rng.choice([x for x in range(1, r1) if c % (r1 - x) == 0] or [0])
        return (
            f"A tank holds {c} litres. It is filled at {r1} litres per minute while a leak drains {r2} litres "
            f"per minute. Starting empty, after how many minutes is it full?",
            c / (r1 - r2),
        )
    if kind == 3:
        a, x = rng.randint(2, 9), rng.randint(-9, 15)
        b = rng.randint(-20, 20)
        c = a * x + b
        k = rng.randint(2, 5)
        return f"If {a}x + {b} = {c}, what is {k}x - 3?", k * x - 3
    if kind == 4:
        p, y = rng.choice([5, 12, 15, 35, 40, 65]), rng.choice([80, 240, 360, 840, 1200])
        q = rng.choice([10, 20, 50])
        return f"What is {p}% of {y}, increased by {q}%?", p * y / 100 * (1 + q / 100)
    d1, t1, d2, t2 = rng.randint(60, 200), rng.randint(1, 3), rng.randint(60, 200), rng.randint(1, 3)
    return (
        f"A car drives {d1} km in {t1} hours and then {d2} km in {t2} hours. What is its average speed in km/h "
        f"over the whole trip? Round to two decimal places.",
        round((d1 + d2) / (t1 + t2), 2),
    )


def _excluded(split: Split, build: object) -> set[str]:
    """Prompts of the calibration split, which the held-out split must never contain."""
    if split == "calibration":
        return set()
    return {t.prompt for t in build(split="calibration")}  # type: ignore[operator]


def math_suite(n: int = 100, split: Split = "calibration") -> list[CalibrationTask]:
    rng = random.Random(_SEEDS[split])
    banned = _excluded(split, math_suite)
    out: list[CalibrationTask] = []
    seen: set[str] = set()
    while len(out) < n:
        question, answer = _math_problem(rng)
        prompt = question + " End your answer with the final number only."
        if prompt in banned or prompt in seen:
            continue
        seen.add(prompt)
        out.append(
            CalibrationTask(
                f"math-{split}-{len(out):03d}", "math", prompt, ReferenceAnswer(answer, rel_tol=1e-3)
            )
        )
    return out


# ------------------------------------------------------------------ structured output
_NAMES = ["Aisha Khan", "Ben Okafor", "Chen Wei", "Dana Levi", "Elif Demir", "Farah Haddad", "Goran Petrov",
          "Hana Sato", "Ivan Novak", "Julia Rossi", "Kofi Mensah", "Lena Fischer", "Mateo Silva", "Nora Berg"]  # fmt: skip
_CITIES = ["Lisbon", "Nairobi", "Osaka", "Toronto", "Krakow", "Lima", "Hanoi", "Dublin", "Accra", "Perth"]
_JOBS = ["nurse", "pilot", "teacher", "chemist", "carpenter", "architect", "baker", "lawyer"]


def json_suite(n: int = 100, split: Split = "calibration") -> list[CalibrationTask]:
    rng = random.Random(_SEEDS[split] + 1)
    banned = _excluded(split, json_suite)
    out: list[CalibrationTask] = []
    seen: set[str] = set()
    while len(out) < n:
        i = len(out)
        name, city, job = rng.choice(_NAMES), rng.choice(_CITIES), rng.choice(_JOBS)
        age, year = rng.randint(21, 70), rng.randint(1995, 2025)
        sentence = rng.choice([
            f"After years abroad, {name}, now {age}, works as a {job} and settled in {city} in {year}.",
            f"In {year} the {age}-year-old {job} {name} moved to {city}.",
            f"{name} ({age}) has lived in {city} since {year}, working as a {job}.",
        ])  # fmt: skip
        prompt = (
            "Extract the person's name, age, job, city and the year they moved there from the text below. "
            'Return only a JSON object with the keys "name", "age", "job", "city" and "year".\n\n'
            f"Text: {sentence}"
        )
        if prompt in banned or prompt in seen:
            continue
        seen.add(prompt)
        expected = (("name", name), ("age", age), ("job", job), ("city", city), ("year", year))
        out.append(
            CalibrationTask(f"json-{split}-{i:03d}", "structured_output", prompt, JsonFields(expected))
        )
    return out


# ------------------------------------------------------------------ code generation
_CODE = [
    ("is_prime", "is_prime(n) that returns True if the integer n is prime",
     "assert [n for n in range(30) if f(n)] == [2,3,5,7,11,13,17,19,23,29]\n    assert f(-7) is False and f(7919)"),
    ("reverse_words", "reverse_words(s) that reverses the order of words in a sentence, collapsing whitespace",
     "assert f('  hello   world ') == 'world hello'\n    assert f('a') == 'a' and f('') == ''"),
    ("fizzbuzz", "fizzbuzz(n) that returns a list of strings for 1..n using the FizzBuzz rules",
     "r = f(15)\n    assert r[2] == 'Fizz' and r[4] == 'Buzz' and r[14] == 'FizzBuzz' and r[0] == '1' and len(r) == 15"),
    ("flatten", "flatten(xs) that flattens an arbitrarily nested list of integers into a flat list",
     "assert f([1,[2,[3,[4]]],5]) == [1,2,3,4,5] and f([]) == [] and f([[[]]]) == []"),
    ("is_palindrome", "is_palindrome(s) that ignores case, spaces and punctuation",
     "assert f('A man, a plan, a canal: Panama') and not f('hello') and f('')"),
    ("merge_sorted", "merge_sorted(a, b) that merges two sorted lists into one sorted list without calling sort",
     "assert f([1,4,9],[2,3,10,11]) == [1,2,3,4,9,10,11] and f([],[1]) == [1]"),
    ("roman_to_int", "roman_to_int(s) that converts a Roman numeral string to an integer",
     "assert f('MCMXCIV') == 1994 and f('III') == 3 and f('XLII') == 42"),
    ("int_to_roman", "int_to_roman(n) that converts an integer from 1 to 3999 to a Roman numeral",
     "assert f(1994) == 'MCMXCIV' and f(4) == 'IV' and f(3999) == 'MMMCMXCIX'"),
    ("anagrams", "anagrams(a, b) that returns True if the two strings are anagrams, ignoring case and spaces",
     "assert f('Dormitory', 'dirty room') and not f('abc', 'abd')"),
    ("gcd", "gcd(a, b) that returns the greatest common divisor of two non-negative integers",
     "assert f(48, 18) == 6 and f(0, 5) == 5 and f(17, 13) == 1"),
    ("run_length", "run_length(s) that run-length encodes a string, e.g. 'aaabcc' -> 'a3b1c2'",
     "assert f('aaabcc') == 'a3b1c2' and f('') == '' and f('z') == 'z1'"),
    ("chunk", "chunk(xs, n) that splits a list into consecutive chunks of size n (the last may be shorter)",
     "assert f([1,2,3,4,5], 2) == [[1,2],[3,4],[5]] and f([], 3) == []"),
    ("second_largest", "second_largest(xs) that returns the second largest distinct value, or None",
     "assert f([3,1,4,4,2]) == 3 and f([5,5]) is None and f([]) is None"),
    ("word_count", "word_count(s) that returns a dict of lowercase word frequencies, ignoring punctuation",
     "assert f('The cat, the hat!') == {'the': 2, 'cat': 1, 'hat': 1}"),
    ("binary_search", "binary_search(xs, x) that returns the index of x in the sorted list xs, or -1",
     "assert f([1,3,5,7,9], 7) == 3 and f([1,3,5], 4) == -1 and f([], 1) == -1"),
    ("transpose", "transpose(m) that transposes a rectangular matrix given as a list of lists",
     "assert f([[1,2,3],[4,5,6]]) == [[1,4],[2,5],[3,6]] and f([]) == []"),
    ("caesar", "caesar(s, k) that shifts letters by k positions, preserving case and leaving other characters",
     "assert f('Hello, World!', 3) == 'Khoor, Zruog!' and f('abc', -1) == 'zab'"),
    ("valid_parens", "valid_parens(s) that checks whether (), [] and {} brackets are balanced and properly nested",
     "assert f('([]{})') and not f('([)]') and f('') and not f('((')"),
    ("digits_sum", "digits_sum(n) that repeatedly sums the digits of a non-negative integer until one digit remains",
     "assert f(9875) == 2 and f(0) == 0 and f(5) == 5"),
    ("dedupe", "dedupe(xs) that removes duplicates from a list while keeping the first occurrence order",
     "assert f([3,1,3,2,1]) == [3,1,2] and f([]) == []"),
    ("rotate", "rotate(xs, k) that rotates a list to the right by k steps (k may exceed the length)",
     "assert f([1,2,3,4,5], 2) == [4,5,1,2,3] and f([1,2,3], 4) == [3,1,2] and f([], 3) == []"),
    ("median", "median(xs) that returns the median of a non-empty list of numbers",
     "assert f([3,1,2]) == 2 and f([4,1,3,2]) == 2.5"),
    ("title_case", "title_case(s) that capitalises the first letter of each word and lowercases the rest",
     "assert f('hELLO wORLD') == 'Hello World' and f('') == ''"),
    ("pascal_row", "pascal_row(n) that returns row n (0-based) of Pascal's triangle as a list",
     "assert f(0) == [1] and f(4) == [1,4,6,4,1]"),
    ("max_subarray", "max_subarray(xs) that returns the largest sum of a contiguous non-empty subarray",
     "assert f([-2,1,-3,4,-1,2,1,-5,4]) == 6 and f([-3,-1,-2]) == -1"),
    ("to_snake", "to_snake(s) that converts a camelCase or PascalCase identifier to snake_case",
     "assert f('camelCaseString') == 'camel_case_string' and f('HTTPServer') == 'http_server' and f('x') == 'x'"),
    ("count_islands", "count_islands(grid) that counts 4-connected groups of 1s in a grid of 0s and 1s",
     "assert f([[1,1,0],[0,1,0],[1,0,1]]) == 3 and f([]) == 0"),
    ("lcs_length", "lcs_length(a, b) that returns the length of the longest common subsequence of two strings",
     "assert f('ABCBDAB', 'BDCABA') == 4 and f('', 'abc') == 0"),
    ("primes_upto", "primes_upto(n) that returns all primes <= n using the sieve of Eratosthenes",
     "assert f(20) == [2,3,5,7,11,13,17,19] and f(1) == []"),
    ("parse_duration", "parse_duration(s) that converts strings like '1h30m', '45s' or '2h5s' into total seconds",
     "assert f('1h30m') == 5400 and f('45s') == 45 and f('2h5s') == 7205"),
]  # fmt: skip


def code_suite() -> list[CalibrationTask]:
    out = []
    for i, (entry, spec, asserts) in enumerate(_CODE):
        prompt = f"Write a Python function {spec}. Reply with one fenced python code block."
        test = f"def check(f):\n    {asserts}\n"
        out.append(CalibrationTask(f"code-{i:03d}", "code_generation", prompt, UnitTests(test, entry)))
    return out


# ------------------------------------------------------------------ instruction following
_TOPICS = {
    "cycling to work": ["helmet", "route", "commute", "traffic", "fitness", "weather"],
    "learning a new language": ["vocabulary", "practice", "grammar", "listening", "habit", "mistakes"],
    "saving money on groceries": ["budget", "list", "discount", "season", "bulk", "waste"],
    "writing a cover letter": ["company", "achievement", "tone", "role", "proofread", "evidence"],
    "running a team meeting": ["agenda", "decision", "timebox", "notes", "owner", "follow-up"],
    "keeping houseplants alive": ["light", "watering", "drainage", "humidity", "soil", "pests"],
    "preparing for a job interview": ["research", "questions", "examples", "posture", "salary", "rehearse"],
    "reducing screen time": ["notifications", "bedtime", "boundaries", "hobbies", "tracking", "walks"],
}
_AVOID = ["very", "really", "important", "always", "just", "make", "good", "time"]


def instructions_suite(n: int = 60, split: Split = "calibration") -> list[CalibrationTask]:
    rng = random.Random(_SEEDS[split] + 2)
    banned = _excluded(split, instructions_suite)
    out: list[CalibrationTask] = []
    seen: set[str] = set()
    while len(out) < n:
        topic = rng.choice(sorted(_TOPICS))
        k, words = rng.randint(3, 5), rng.randint(6, 9)
        w1, w2 = rng.sample(_TOPICS[topic], 2)
        i1, i2 = rng.sample(range(1, k + 1), 2)
        avoid = rng.choice(_AVOID)
        end = rng.choice(["END", "DONE"])
        prompt = (
            f"Write tips about {topic}. Rules: give exactly {k} bullet points, each on its own line starting with "
            f'"- "; every bullet has exactly {words} words; bullet {i1} must contain the word "{w1}" and bullet '
            f'{i2} the word "{w2}"; never use the word "{avoid}"; after the bullets write one final line '
            f"containing only {end}. Output nothing else."
        )
        if prompt in banned or prompt in seen:
            continue
        seen.add(prompt)
        check = BulletConstraints(k, words, ((w1, i1), (w2, i2)), (avoid,), end)
        out.append(CalibrationTask(f"instr-{split}-{len(out):03d}", "content_generation", prompt, check))
    return out


def suite(category: str, split: Split = "calibration") -> list[CalibrationTask]:
    if category == "math":
        return math_suite(split=split)
    if category == "structured_output":
        return json_suite(split=split)
    if category == "instructions":
        return instructions_suite(split=split)
    if category == "code_generation":
        if split != "calibration":
            raise ValueError("the code suite has no held-out split yet")
        return code_suite()
    raise ValueError(f"no calibration suite for category {category!r}")


SUITE_CATEGORIES = ("math", "structured_output", "instructions", "code_generation")
