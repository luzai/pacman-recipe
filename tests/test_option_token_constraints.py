from __future__ import annotations

import unittest

from pacman_recipe.level1.token_constraints import (
    ObjectiveParseError,
    ObjectiveTokenConstraint,
)


class FakeTokenizer:
    def encode(self, text, **_):
        return [ord(character) for character in text]

    def decode(self, ids, **_):
        return "".join(chr(item) for item in ids)


class ObjectiveTokenConstraintTests(unittest.TestCase):
    def test_option_codes_are_one_token_exact_and_auditable(self) -> None:
        tokenizer = FakeTokenizer()
        constraint = ObjectiveTokenConstraint.build(
            tokenizer, ["C0", "C1", "A0", "E0"]
        )
        self.assertEqual(constraint.rendered_choices, ("B", "F", "J", "P"))
        self.assertEqual(constraint.max_new_tokens, 1)
        self.assertEqual(
            constraint.allowed_token_ids,
            [ord(code) for code in "BFJP"],
        )
        self.assertEqual(constraint.code_for_option("C1"), "F")
        self.assertEqual(constraint.option_for_completion("F"), "C1")
        self.assertEqual(constraint.option_for_tokens([ord("F")]), "C1")
        self.assertEqual(
            constraint.support_ledger([ord("F")]),
            [[ord(code) for code in "BFJP"]],
        )

    def test_invalid_code_completion_and_tokens_are_rejected(self) -> None:
        constraint = ObjectiveTokenConstraint.build(FakeTokenizer(), ["C0", "A0"])
        for completion in ("C0", " B", "B\n", "Z"):
            with self.subTest(completion=completion), self.assertRaises(
                ObjectiveParseError
            ):
                constraint.option_for_completion(completion)
        for tokens in ([], [ord("Z")], [ord("B"), ord("F")]):
            with self.subTest(tokens=tokens), self.assertRaises(ObjectiveParseError):
                constraint.option_for_tokens(tokens)

    def test_tokenizer_contract_and_alphabet_limit_fail_closed(self) -> None:
        class MultiTokenTokenizer(FakeTokenizer):
            def encode(self, text, **_):
                return [1, 2]

            def decode(self, ids, **_):
                return "B"

        with self.assertRaisesRegex(ValueError, "one token"):
            ObjectiveTokenConstraint.build(MultiTokenTokenizer(), ["C0"])

        class SpecialTokenTokenizer(FakeTokenizer):
            all_special_ids = [ord("B")]

        with self.assertRaisesRegex(ValueError, "special token"):
            ObjectiveTokenConstraint.build(SpecialTokenTokenizer(), ["C0"])
        with self.assertRaisesRegex(ValueError, "unsupported option IDs"):
            ObjectiveTokenConstraint.build(
                FakeTokenizer(),
                ["C0", "C4"],
            )


if __name__ == "__main__":
    unittest.main()
