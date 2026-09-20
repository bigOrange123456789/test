"""共享生成长度策略的上下限、缩放与参数校验测试。"""

from __future__ import annotations

import unittest

from script.lib.token_budget import (
    FACT_LENGTH_BUDGET_DEFAULTS, LENGTH_BUDGET_DEFAULTS,
    calculate_token_budget, normalize_length_budget,
)


class TokenBudgetTests(unittest.TestCase):
    def test_default_policies_are_independent_and_can_be_partially_overridden(self):
        policy = normalize_length_budget(None, "generation.lengthBudget")
        self.assertEqual(policy, LENGTH_BUDGET_DEFAULTS)
        policy["enabled"] = True
        self.assertFalse(LENGTH_BUDGET_DEFAULTS["enabled"])
        fact_policy = normalize_length_budget({"enabled": True}, "factScore.lengthBudget", defaults=FACT_LENGTH_BUDGET_DEFAULTS)
        self.assertEqual(fact_policy, {**FACT_LENGTH_BUDGET_DEFAULTS, "enabled": True})
        partial_defaults = normalize_length_budget(None, "budget", defaults={"multiplier": 3})
        self.assertEqual(partial_defaults["multiplier"], 3)
        self.assertEqual(partial_defaults["maxNewTokens"], 4096)

    def test_scales_rounds_up_and_clamps_to_both_bounds(self):
        policy = {"multiplier": 1.5, "extraTokens": 3, "minNewTokens": 10, "maxNewTokens": 20}
        self.assertEqual(calculate_token_budget(0, policy), 10)
        self.assertEqual(calculate_token_budget(5, policy), 11)
        self.assertEqual(calculate_token_budget(8, policy), 15)
        self.assertEqual(calculate_token_budget(100, policy), 20)
        self.assertEqual(calculate_token_budget(10 ** 400, policy), 20)

    def test_rejects_invalid_or_unknown_policy_fields_even_when_disabled(self):
        invalid = [False, [], "auto", {"surprise": 1}, {"enabled": 1}, {"retryOnTruncation": "true"},
                   {"multiplier": True}, {"multiplier": "2"}, {"multiplier": 0}, {"multiplier": -1},
                   {"multiplier": 33}, {"multiplier": float("inf")}, {"multiplier": float("nan")},
                   {"extraTokens": -1}, {"extraTokens": True}, {"extraTokens": 32769},
                   {"minNewTokens": 0}, {"minNewTokens": 1.5}, {"maxNewTokens": 32769},
                   {"minNewTokens": 9000, "maxNewTokens": 8192}]
        for value in invalid:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "testBudget"):
                normalize_length_budget(value, "testBudget")
        with self.assertRaises(ValueError):
            normalize_length_budget(None, "testBudget", defaults={"unknown": 1})

    def test_count_requires_nonnegative_integer(self):
        for count in (True, False, -1, 1.0, "10", None):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "token_count"):
                calculate_token_budget(count, LENGTH_BUDGET_DEFAULTS)


if __name__ == "__main__":
    unittest.main()
