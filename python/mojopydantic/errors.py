from __future__ import annotations

import json
from typing import Any


_MESSAGES = {
    "missing": "Field required",
    "extra_forbidden": "Extra inputs are not permitted",
    "int_type": "Input should be a valid integer",
    "int_parsing": "Input should be a valid integer, unable to parse string as an integer",
    "int_from_float": "Input should be a valid integer, got a number with a fractional part",
    "float_type": "Input should be a valid number",
    "float_parsing": "Input should be a valid number, unable to parse string as a number",
    "bool_type": "Input should be a valid boolean",
    "bool_parsing": "Input should be a valid boolean, unable to interpret input",
    "string_type": "Input should be a valid string",
    "bytes_type": "Input should be a valid bytes",
    "list_type": "Input should be a valid list",
    "tuple_type": "Input should be a valid tuple",
    "set_type": "Input should be a valid set",
    "dict_type": "Input should be a valid dictionary",
    "model_type": "Input should be a valid dictionary or instance of the model",
    "none_required": "Input should be None",
    "literal_error": "Input should be an accepted literal",
    "enum": "Input should be a valid enumeration member",
    "greater_than": "Input should be greater than {gt}",
    "greater_than_equal": "Input should be greater than or equal to {ge}",
    "less_than": "Input should be less than {lt}",
    "less_than_equal": "Input should be less than or equal to {le}",
    "too_short": "Value should have at least {min_length} items",
    "too_long": "Value should have at most {max_length} items",
    "string_pattern_mismatch": "String should match pattern {pattern!r}",
    "value_error": "Value error, {error}",
    "assertion_error": "Assertion failed, {error}",
    "json_invalid": "Invalid JSON: {error}",
    "date_from_datetime_parsing": "Input should be a valid date",
    "datetime_from_date_parsing": "Input should be a valid datetime",
    "uuid_parsing": "Input should be a valid UUID",
    "decimal_parsing": "Input should be a valid decimal",
}


def issue(
    kind: str,
    loc: tuple[Any, ...],
    value: Any,
    *,
    msg: str | None = None,
    ctx: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = ctx or {}
    template = msg or _MESSAGES.get(kind, "Input should be valid")
    try:
        message = template.format(**context)
    except (KeyError, ValueError):
        message = template
    result: dict[str, Any] = {"type": kind, "loc": loc, "msg": message, "input": value}
    if context:
        result["ctx"] = context
    return result


class ValidationError(ValueError):
    def __init__(self, title: str, line_errors: list[dict[str, Any]]) -> None:
        self.title = title
        self._errors = line_errors
        super().__init__(str(self))

    @classmethod
    def from_exception_data(
        cls, title: str, line_errors: list[dict[str, Any]], input_type: str = "python"
    ) -> "ValidationError":
        return cls(title, line_errors)

    def error_count(self) -> int:
        return len(self._errors)

    def errors(
        self,
        *,
        include_url: bool = True,
        include_context: bool = True,
        include_input: bool = True,
    ) -> list[dict[str, Any]]:
        output = []
        for original in self._errors:
            item = dict(original)
            if not include_context:
                item.pop("ctx", None)
            if not include_input:
                item.pop("input", None)
            if include_url:
                item["url"] = f"https://errors.pydantic.dev/2/v/{item['type']}"
            output.append(item)
        return output

    def json(
        self,
        *,
        indent: int | None = None,
        include_url: bool = True,
        include_context: bool = True,
        include_input: bool = True,
    ) -> str:
        return json.dumps(
            self.errors(
                include_url=include_url,
                include_context=include_context,
                include_input=include_input,
            ),
            indent=indent,
            default=str,
        )

    def __str__(self) -> str:
        count = len(self._errors)
        heading = f"{count} validation error{'s' if count != 1 else ''} for {self.title}"
        lines = [heading]
        for item in self._errors:
            location = ".".join(str(part) for part in item["loc"]) or "__root__"
            lines.append(f"{location}\n  {item['msg']} [type={item['type']}]")
        return "\n".join(lines)


class PydanticUserError(RuntimeError):
    def __init__(self, message: str, *, code: str | None = None) -> None:
        self.code = code
        super().__init__(message)


class PydanticSchemaGenerationError(PydanticUserError):
    pass
