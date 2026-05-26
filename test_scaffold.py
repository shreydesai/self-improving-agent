"""
Self-contained test suite — no live API key required.
Mocks the Anthropic client to verify the full loop.
"""
from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch
sys.path.insert(0, ".")


# ── helpers to build fake Anthropic responses ──────────────────────────────────

def _make_text_block(text: str):
    b = MagicMock()
    b.type = "text"
    b.text = text
    return b


def _make_tool_use_block(name: str, input_: dict, tool_id: str = "tu_1"):
    b = MagicMock()
    b.type = "tool_use"
    b.name = name
    b.input = input_
    b.id = tool_id
    return b


def _make_response(stop_reason: str, *blocks):
    r = MagicMock()
    r.stop_reason = stop_reason
    r.content = list(blocks)
    r.usage = MagicMock()
    r.usage.input_tokens = 100
    r.usage.output_tokens = 50
    return r


# ── test 1: tools module ───────────────────────────────────────────────────────

def test_tools():
    from tools import calculator, web_search, python_exec, memory_write, memory_read, memory_clear

    memory_clear()

    assert calculator("2 + 2") == "4", "basic arithmetic"
    assert calculator("sqrt(144)") == "12.0", "sqrt"
    assert calculator("15 * 0.1 + 30") == "31.5", "mixed"
    assert "Error" in calculator("__import__('os')"), "injection blocked"

    r = web_search("boiling point of water fahrenheit")
    assert "212" in r, f"web_search failed: {r!r}"

    r = web_search("chemical symbol for gold")
    assert "Au" in r, f"web_search Au failed: {r!r}"

    r = web_search("this query has no results xyz123")
    assert "No results" in r

    r = python_exec("print(sum(range(1, 11)))")
    assert r == "55", f"python_exec sum failed: {r!r}"

    r = python_exec("x = 1/0")
    assert "Error" in r or "ZeroDivisionError" in r

    assert memory_write("k", "v") == "Stored 'k'"
    assert memory_read("k") == "v"
    assert "not found" in memory_read("missing")

    print("✓ test_tools passed")


# ── test 2: evaluator ──────────────────────────────────────────────────────────

def test_evaluator():
    from tasks import evaluate

    assert evaluate("66.0", "66.0")
    assert evaluate("The answer is 66.0.", "66.0")
    assert evaluate("66", "66.0")          # numeric match
    assert evaluate("result is 77", "77")
    assert evaluate("0, 1, 1, 2, 3, 5, 8, 13", "0, 1, 1, 2, 3, 5, 8, 13")
    assert evaluate("sequence: 0, 1, 1, 2, 3, 5, 8, 13 as computed", "0, 1, 1, 2, 3, 5, 8, 13")
    assert not evaluate("42", "99")
    assert evaluate("Jupiter is the largest planet", "Jupiter")
    assert evaluate("The answer is Au", "Au")

    print("✓ test_evaluator passed")


# ── test 3: scaffold versioning ────────────────────────────────────────────────

def test_scaffold():
    from scaffold import make_initial_scaffold, snapshot, load_scaffold

    s = make_initial_scaffold()
    assert len(s.tools) == 5
    assert any(t.name == "calculator" for t in s.tools)

    s = snapshot(s, generation=0, rationale="test")
    s2 = load_scaffold(s.version_id)
    assert s2.version_id == s.version_id
    assert s2.generation == 0
    assert len(s2.tools) == 5

    print("✓ test_scaffold passed")


# ── test 4: mutation application ──────────────────────────────────────────────

def test_mutations():
    from scaffold import make_initial_scaffold, snapshot
    from meta_agent import apply_mutation, MutationProposal

    s = make_initial_scaffold()
    s = snapshot(s, 0)

    # add_tool
    p = MutationProposal(
        mutation_type="add_tool",
        rationale="test add",
        details={
            "name": "date_tool",
            "description": "returns today's date",
            "parameters": {"type": "object", "properties": {}, "required": []},
            "implementation": "def date_tool():\n    import datetime\n    return str(datetime.date.today())",
        },
    )
    s2 = apply_mutation(s, p)
    assert len(s2.tools) == 6
    assert any(t.name == "date_tool" for t in s2.tools)
    assert s2.parent_id == s.version_id

    # duplicate add_tool → no-op
    s3 = apply_mutation(s2, p)
    assert len(s3.tools) == 6

    # rewrite_planning_policy
    p2 = MutationProposal(
        mutation_type="rewrite_planning_policy",
        rationale="better planning",
        details={"new_system_prompt": "New policy content here."},
    )
    s4 = apply_mutation(s2, p2)
    assert s4.planning_policy.system_prompt == "New policy content here."

    # add_routing_rule
    p3 = MutationProposal(
        mutation_type="add_routing_rule",
        rationale="route date tasks",
        details={"condition": "if task asks about current date", "preferred_tools": ["date_tool"], "priority": 5},
    )
    s5 = apply_mutation(s4, p3)
    assert len(s5.router_rules) == 1

    # remove non-protected tool
    p4 = MutationProposal(
        mutation_type="remove_tool",
        rationale="cleanup",
        details={"name": "date_tool"},
    )
    s6 = apply_mutation(s5, p4)
    assert not any(t.name == "date_tool" for t in s6.tools)

    # remove protected tool → no-op
    p5 = MutationProposal(
        mutation_type="remove_tool",
        rationale="try to remove core tool",
        details={"name": "calculator"},
    )
    s7 = apply_mutation(s5, p5)
    assert any(t.name == "calculator" for t in s7.tools)

    print("✓ test_mutations passed")


# ── test 5: agent episode with mocked client ──────────────────────────────────

def test_agent_episode():
    from scaffold import make_initial_scaffold, snapshot
    from agent import run_episode
    from tasks import evaluate

    s = make_initial_scaffold()
    s = snapshot(s, 0)

    # Scenario: agent calls calculator, then returns answer
    call_count = [0]

    def fake_create(**kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            # First call: use calculator
            return _make_response(
                "tool_use",
                _make_text_block("I'll use the calculator."),
                _make_tool_use_block("calculator", {"expression": "15 * 0.15 + 30"}, "tu_1"),
            )
        else:
            # Second call: final answer
            return _make_response("end_turn", _make_text_block("The result is 32.25"))

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = fake_create

    traj = run_episode(
        task_id="test_mock",
        task_text="What is 15% of 15, plus 30?",
        correct_answer="32.25",
        scaffold=s,
        client=mock_client,
        evaluator=evaluate,
    )

    assert traj.success, f"Expected success, got answer={traj.final_answer!r}"
    assert len(traj.steps) >= 1
    tool_step = next(st for st in traj.steps if st.tool_name == "calculator")
    assert "32.25" in tool_step.tool_output, f"unexpected tool output: {tool_step.tool_output}"
    assert traj.tokens_used > 0

    print(f"✓ test_agent_episode passed  (steps={len(traj.steps)}, answer={traj.final_answer!r})")


# ── test 6: dynamic tool execution ────────────────────────────────────────────

def test_dynamic_tool():
    from tools import run_dynamic_tool

    impl = """
def levenshtein(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]
"""
    result = run_dynamic_tool(impl, "levenshtein", {"s1": "kitten", "s2": "sitting"})
    assert result == "3", f"Expected '3', got {result!r}"
    print(f"✓ test_dynamic_tool passed  (levenshtein='kitten','sitting' → {result})")


# ── run all ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_tools,
        test_evaluator,
        test_scaffold,
        test_mutations,
        test_agent_episode,
        test_dynamic_tool,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"✗ {t.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {t.__name__} ERROR: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            failed += 1

    print(f"\n{'All tests passed' if failed == 0 else f'{failed} test(s) failed'}")
    sys.exit(0 if failed == 0 else 1)
