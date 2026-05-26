"""
Task suite (30 tasks) + evaluator.
Split: 18 train / 6 val / 6 test (fixed seed ordering).
Tasks designed to start ~45% solvable with the initial scaffold.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Literal, Optional

Category = Literal["math", "code", "qa", "multi_step", "hard"]


@dataclass
class Task:
    id: str
    text: str
    correct_answer: str
    category: Category
    split: Literal["train", "val", "test"]
    notes: str = ""


# ── task definitions ───────────────────────────────────────────────────────────

ALL_TASKS: List[Task] = [

    # ── MATH (calculator-solvable) ────────────────────────────────────────────
    Task("math_01", "Calculate 15% of 240, then add 30. What is the result?",
         "66.0", "math", "train",
         "0.15 * 240 + 30 = 36 + 30 = 66"),
    Task("math_02", "A rectangle has length 12.5 and width 8.3. What is its area?",
         "103.75", "math", "train",
         "12.5 * 8.3"),
    Task("math_03", "What is the sum of all prime numbers less than 20?",
         "77", "math", "train",
         "2+3+5+7+11+13+17+19 = 77"),
    Task("math_04", "A train travels 150 miles at 60 mph, then 120 miles at 80 mph. "
         "What is the total travel time in hours?",
         "4.0", "math", "train",
         "150/60 + 120/80 = 2.5 + 1.5 = 4.0"),
    Task("math_05", "Convert 98.6 degrees Fahrenheit to Celsius. Use C = (F - 32) * 5 / 9. "
         "Round to 1 decimal place.",
         "37.0", "math", "train",
         "(98.6-32)*5/9 = 37.0"),
    Task("math_06", "A store sells 5 items at $12.50 each and 3 items at $8.75 each. "
         "What is the total revenue?",
         "88.75", "math", "train",
         "5*12.5 + 3*8.75 = 62.5 + 26.25 = 88.75"),

    # ── CODE (python_exec-solvable) ───────────────────────────────────────────
    Task("code_01", "Write and execute Python code to compute the 10th Fibonacci number. "
         "Use 0-indexing: F(0)=0, F(1)=1, F(2)=1, ...",
         "55", "code", "train",
         "F(10) = 55"),
    Task("code_02", "Write Python code to count the number of vowels (a, e, i, o, u) "
         "in the string 'Hello World'. Print the count.",
         "3", "code", "train",
         "e, o, o → 3"),
    Task("code_03", "Write Python code to compute the GCD of 48 and 18. Print the result.",
         "6", "code", "train",
         "GCD(48,18) = 6"),
    Task("code_04", "Write Python code to reverse the string 'anthropic' and print the result.",
         "cipohtrna", "code", "train"),
    Task("code_05", "Write Python code to compute 8 factorial (8!) and print the result.",
         "40320", "code", "train"),
    Task("code_06", "Write Python code to find and print the maximum value "
         "in the list [3, 7, 2, 9, 1, 5].",
         "9", "code", "train"),

    # ── QA (web_search-solvable) ──────────────────────────────────────────────
    Task("qa_01", "How many days are in a leap year?",
         "366", "qa", "train"),
    Task("qa_02", "What is the boiling point of water in degrees Fahrenheit?",
         "212", "qa", "train"),
    Task("qa_03", "What is the chemical symbol for Gold?",
         "Au", "qa", "train"),
    Task("qa_04", "What are the first 8 numbers in the Fibonacci sequence? "
         "Start from F(0)=0.",
         "0, 1, 1, 2, 3, 5, 8, 13", "qa", "train"),

    # ── HARD (require multi-step or new capabilities) ─────────────────────────
    Task("hard_01", "What is the Levenshtein (edit) distance between the strings "
         "'kitten' and 'sitting'? "
         "Count the minimum number of single-character edits (insertions, deletions, substitutions).",
         "3", "hard", "train",
         "kitten→sitten(sub k→s), sitten→sittin(sub e→i), sittin→sitting(insert g): 3 ops"),
    Task("hard_02", "What is the median of the list [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]? "
         "Sort the list first, then find the middle value.",
         "3.5", "hard", "train",
         "sorted=[1,1,2,3,3,4,5,5,6,9], n=10, median=(3+4)/2=3.5"),

    # ── VAL ───────────────────────────────────────────────────────────────────
    Task("val_math_01", "What is the area of a circle with radius 7? "
         "Use pi = 3.14159265358979. Round to 2 decimal places.",
         "153.94", "math", "val",
         "pi * 49 = 153.938..."),
    Task("val_math_02", "Starting with $1000, apply 5% annual compound interest for 3 years. "
         "Compute P*(1+r)^t where P=1000, r=0.05, t=3. Round to 2 decimal places.",
         "1157.63", "math", "val",
         "1000 * 1.05^3 = 1157.625"),
    Task("val_code_01", "Write Python code to encode the string 'hello world' "
         "using base64 encoding. Print the result.",
         "aGVsbG8gd29ybGQ=", "code", "val"),
    Task("val_code_02", "Write Python code to check whether 997 is a prime number. "
         "Print True or False.",
         "True", "code", "val"),
    Task("val_qa_01", "How many bytes are in a kilobyte?",
         "1024", "qa", "val"),
    Task("val_hard_01", "Generate the Collatz sequence starting from 6. "
         "If the number is even divide by 2, if odd multiply by 3 and add 1. "
         "Stop when you reach 1. List all numbers in order including 6 and 1.",
         "6, 3, 10, 5, 16, 8, 4, 2, 1", "hard", "val",
         "6→3→10→5→16→8→4→2→1"),

    # ── TEST (held out) ───────────────────────────────────────────────────────
    Task("test_math_01", "A car costs $25000 new and depreciates 15% per year. "
         "What is it worth after 3 years? Round to 2 decimal places.",
         "15353.13", "math", "test",
         "25000 * 0.85^3 = 15353.125"),
    Task("test_math_02", "What is (17 * 23) - (45 / 9) + 12?",
         "398.0", "math", "test",
         "391 - 5 + 12 = 398"),
    Task("test_code_01", "Write Python code to find all factors of 36, "
         "sorted in ascending order. Print them as a list.",
         "[1, 2, 3, 4, 6, 9, 12, 18, 36]", "code", "test"),
    Task("test_code_02", "Write Python code to convert the hexadecimal string 'FF' "
         "to its decimal equivalent. Print the result.",
         "255", "code", "test"),
    Task("test_qa_01", "What is the largest planet in our solar system?",
         "Jupiter", "qa", "test"),
    Task("test_hard_01", "What day of the week was January 1, 2025? "
         "Use computation or look it up.",
         "Wednesday", "hard", "test",
         "Jan 1 2024 = Monday, 2024 is leap year (366 days = 52w+2), so Jan 1 2025 = Wednesday"),
]

assert len(ALL_TASKS) == 30, f"Expected 30 tasks, got {len(ALL_TASKS)}"

TRAIN_TASKS = [t for t in ALL_TASKS if t.split == "train"]
VAL_TASKS   = [t for t in ALL_TASKS if t.split == "val"]
TEST_TASKS  = [t for t in ALL_TASKS if t.split == "test"]

assert len(TRAIN_TASKS) == 18
assert len(VAL_TASKS) == 6
assert len(TEST_TASKS) == 6


# ── evaluator ─────────────────────────────────────────────────────────────────

def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(".")
    return s


def _numeric_match(predicted: str, correct: str, tol: float = 1e-6) -> bool:
    try:
        p = float(re.search(r"-?\d+\.?\d*", predicted.replace(",", "")).group())
        c = float(correct.replace(",", ""))
        return abs(p - c) <= tol + abs(c) * tol
    except (AttributeError, ValueError):
        return False


def evaluate(predicted: str, correct: str) -> bool:
    """Return True if predicted answer matches correct answer."""
    pred_n = _normalize(predicted)
    corr_n = _normalize(correct)

    # Exact match after normalization
    if pred_n == corr_n:
        return True

    # Correct answer appears verbatim in prediction
    if corr_n in pred_n:
        return True

    # Numeric match
    if _numeric_match(predicted, correct):
        return True

    # List match: check all required elements present
    corr_parts = [p.strip() for p in correct.split(",")]
    if len(corr_parts) >= 2:
        if all(_normalize(p) in pred_n for p in corr_parts):
            return True

    return False


def score(trajectories) -> float:
    if not trajectories:
        return 0.0
    return sum(1 for t in trajectories if t.success) / len(trajectories)
