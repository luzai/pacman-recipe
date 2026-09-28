"""Exact single-token constraints for Edward option choices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


EDWARD_OPTION_CONSTRAINT = "edward-option-code-v1"
# Ten deterministic codes cover the planner's maximum 4 C + 4 A + 2 E
# candidates.  Movement letters U/D/L/R and strategy initials C/A/E are
# deliberately excluded so the completion cannot be mistaken for either.
OPTION_CODE_ALPHABET = tuple("BFGHJKMNPQ")
OPTION_IDS = (
    "C0",
    "C1",
    "C2",
    "C3",
    "A0",
    "A1",
    "A2",
    "A3",
    "E0",
    "E1",
)
OPTION_CODE_BY_ID = dict(zip(OPTION_IDS, OPTION_CODE_ALPHABET, strict=True))


class ObjectiveParseError(ValueError):
    """Raised when a completion is not one advertised objective."""


@dataclass(frozen=True)
class ObjectiveTokenConstraint:
    """Map advertised planner options to exact one-token policy actions."""

    option_ids: tuple[str, ...]
    rendered_choices: tuple[str, ...]
    token_sequences: tuple[tuple[int, ...], ...]

    @classmethod
    def build(cls, tokenizer: Any, option_ids: Iterable[str]) -> "ObjectiveTokenConstraint":
        normalized = tuple(dict.fromkeys(str(item) for item in option_ids))
        if not normalized:
            raise ValueError("objective constraint requires at least one option")
        unknown = [item for item in normalized if item not in OPTION_CODE_BY_ID]
        if unknown:
            raise ValueError(
                f"objective constraint has unsupported option IDs: {unknown!r}"
            )
        rendered = tuple(OPTION_CODE_BY_ID[item] for item in normalized)
        special_ids = {
            int(item) for item in getattr(tokenizer, "all_special_ids", ())
        }
        sequences: list[tuple[int, ...]] = []
        for text in rendered:
            token_ids = tuple(
                int(item)
                for item in tokenizer.encode(text, add_special_tokens=False)
            )
            if len(token_ids) != 1:
                raise ValueError(
                    "objective option code must tokenize to one token: "
                    f"{text!r} -> {token_ids!r}"
                )
            if token_ids[0] in special_ids:
                raise ValueError(
                    "objective option code cannot use a special token: "
                    f"{text!r} -> {token_ids!r}"
                )
            decoded = tokenizer.decode(
                list(token_ids),
                skip_special_tokens=True,
            )
            if decoded != text:
                raise ValueError(
                    f"objective choice does not round-trip exactly: {text!r} -> {decoded!r}"
                )
            sequences.append(token_ids)
        if len(set(sequences)) != len(sequences):
            raise ValueError("objective option codes must have unique token IDs")
        return cls(normalized, rendered, tuple(sequences))

    @property
    def max_new_tokens(self) -> int:
        return 1

    @property
    def allowed_token_ids(self) -> list[int]:
        return sorted({token for sequence in self.token_sequences for token in sequence})

    def option_for_tokens(self, output_tokens: Iterable[int]) -> str:
        actual = tuple(int(item) for item in output_tokens)
        try:
            index = self.token_sequences.index(actual)
        except ValueError as exc:
            raise ObjectiveParseError(
                f"output tokens {actual!r} are not one canonical objective"
            ) from exc
        return self.option_ids[index]

    def code_for_option(self, option_id: str) -> str:
        try:
            index = self.option_ids.index(str(option_id))
        except ValueError as exc:
            raise ObjectiveParseError(
                f"option {option_id!r} was not advertised"
            ) from exc
        return self.rendered_choices[index]

    def option_for_completion(self, completion: str) -> str:
        if not isinstance(completion, str):
            raise ObjectiveParseError("objective completion must be a string")
        try:
            index = self.rendered_choices.index(completion)
        except ValueError as exc:
            raise ObjectiveParseError(
                f"completion {completion!r} is not an advertised option code"
            ) from exc
        return self.option_ids[index]

    def support_ledger(self, output_tokens: Iterable[int]) -> list[list[int]]:
        """Return the exact option-token denominator before sampling."""

        actual = tuple(int(item) for item in output_tokens)
        self.option_for_tokens(actual)
        if len(actual) != 1:
            raise ObjectiveParseError("objective output must contain exactly one token")
        return [self.allowed_token_ids]
