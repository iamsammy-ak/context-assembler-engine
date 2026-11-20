"""Tool schemas: OpenAI function format, a relevance document per tool, and lossless-to-the-caller
compression.

Compression keeps everything a model needs to *call* the tool correctly (name, parameter names, types,
enums, required fields, nesting) and strips annotation keywords meant for humans (descriptions,
titles, examples, defaults). Property *names* are never touched, even a property called
``description``.
"""

from __future__ import annotations

from typing import Any

from .schema import ToolSpec
from .text import split_identifier, split_sentences

ANNOTATION_KEYWORDS = frozenset(
    {"description", "title", "examples", "example", "default", "$comment", "deprecated", "readOnly", "writeOnly"}
)
_NAMED_SCHEMAS = frozenset({"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"})
_SUBSCHEMAS = frozenset(
    {"items", "additionalProperties", "not", "contains", "if", "then", "else", "propertyNames",
     "anyOf", "oneOf", "allOf", "prefixItems"}
)  # fmt: skip
MAX_COMPRESSED_DESCRIPTION = 100


def to_openai_tool(spec: ToolSpec) -> dict[str, Any]:
    """The ``{"type": "function", "function": {...}}`` definition sent in the ``tools`` list."""
    function: dict[str, Any] = {"name": spec.name}
    if spec.description:
        function["description"] = spec.description
    function["parameters"] = spec.parameters
    return {"type": "function", "function": function}


def strip_annotations(schema: Any) -> Any:
    """A copy of a JSON Schema without annotation keywords, recursing into sub-schemas only."""
    if isinstance(schema, list):
        return [strip_annotations(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    stripped: dict[str, Any] = {}
    for key, value in schema.items():
        if key in ANNOTATION_KEYWORDS:
            continue
        if key in _NAMED_SCHEMAS and isinstance(value, dict):
            stripped[key] = {name: strip_annotations(sub) for name, sub in value.items()}
        elif key in _SUBSCHEMAS:
            stripped[key] = strip_annotations(value)
        else:
            stripped[key] = value  # type, enum, const, required, format, bounds ... kept verbatim
    return stripped


def short_description(text: str, limit: int = MAX_COMPRESSED_DESCRIPTION) -> str:
    """First sentence of ``text``, cut at a word boundary to at most ``limit`` characters."""
    sentences = split_sentences(text)
    first = sentences[0] if sentences else ""
    if len(first) <= limit:
        return first
    return first[:limit].rsplit(" ", 1)[0].rstrip(",;:") + "…"


def compress_tool(definition: dict[str, Any]) -> dict[str, Any]:
    """A smaller tool definition: one-sentence description and an annotation-free schema."""
    function = definition["function"]
    compact: dict[str, Any] = {"name": function["name"]}
    description = short_description(function.get("description", ""))
    if description:
        compact["description"] = description
    compact["parameters"] = strip_annotations(function.get("parameters", {}))
    return {"type": "function", "function": compact}


def tool_document(definition: dict[str, Any]) -> str:
    """Text used to rank a tool against the query: name words, description and parameter docs."""
    function = definition["function"]
    parts = [split_identifier(function["name"]), function.get("description", "")]
    properties = function.get("parameters", {}).get("properties", {})
    if isinstance(properties, dict):
        for name, schema in properties.items():
            parts.append(split_identifier(name))
            if isinstance(schema, dict) and isinstance(schema.get("description"), str):
                parts.append(schema["description"])
    return " ".join(part for part in parts if part)
