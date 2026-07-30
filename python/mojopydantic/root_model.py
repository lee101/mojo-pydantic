from __future__ import annotations

from typing import Any, ClassVar

from .fields import PydanticUndefined
from .main import BaseModel


class RootModel(BaseModel):
    root: Any
    __pydantic_root_model__: ClassVar[bool] = True
    _specializations: ClassVar[dict[Any, type["RootModel"]]] = {}

    def __init__(self, root: Any = PydanticUndefined, /, **data: Any) -> None:
        if root is not PydanticUndefined:
            if data:
                raise ValueError("RootModel accepts either a root value or keyword data")
            data = {"root": root}
        super().__init__(**data)

    @classmethod
    def __class_getitem__(cls, item: Any) -> type["RootModel"]:
        if item not in cls._specializations:
            name = f"RootModel[{getattr(item, '__name__', str(item))}]"
            cls._specializations[item] = type(
                name,
                (cls,),
                {"__annotations__": {"root": item}, "__module__": cls.__module__},
            )
        return cls._specializations[item]

    @classmethod
    def model_validate(cls, obj: Any, **kwargs: Any) -> "RootModel":
        if isinstance(obj, cls):
            return obj
        return cls._validate_model({"root": obj}, **kwargs)

    @classmethod
    def model_validate_json(cls, json_data: str | bytes | bytearray, **kwargs: Any) -> "RootModel":
        from .type_adapter import TypeAdapter

        value = TypeAdapter(cls.model_fields["root"].annotation).validate_json(
            json_data, strict=kwargs.pop("strict", None), context=kwargs.pop("context", None)
        )
        return cls.model_validate(value, **kwargs)

    def model_dump(self, **kwargs: Any) -> Any:
        mode = kwargs.get("mode", "python")
        from ._core import dump_value

        return dump_value(self.root, mode, bool(kwargs.get("by_alias")))

    def model_dump_json(self, *, indent: int | None = None, **kwargs: Any) -> str:
        import json

        separators = None if indent is not None else (",", ":")
        return json.dumps(
            self.model_dump(mode="json", **kwargs),
            indent=indent,
            separators=separators,
        )

    @classmethod
    def model_json_schema(cls, **kwargs: Any) -> dict[str, Any]:
        from ._core import schema_for

        return schema_for(cls.model_fields["root"].annotation)
