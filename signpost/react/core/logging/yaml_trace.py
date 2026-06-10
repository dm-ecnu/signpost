"""YAML format iteration log module.

Provides structured YAML logging for LLM iteration traces.

Features:
- Structured YAML output
- Long content uses YAML literal style (|-) for readability
- Unicode support
- Key ordering preserved
- Minimal API: single save function
"""

import json
import yaml
from pathlib import Path
from typing import Any, Optional
from datetime import datetime, timezone


class StripLiteralString(str):
    """Marker for strings that should use YAML literal style with strip chomping (|-)

    YAML syntax:
    - `|-` strips all trailing newlines
    - Example: "Line 1\\nLine 2\\n" becomes:
      ```yaml
      key: |-
        Line 1
        Line 2
      ```
    """

    pass


def strip_literal_str_representer(dumper, data):
    """YAML representer for StripLiteralString: forces |- syntax"""
    stripped_data = data.rstrip("\n")
    return dumper.represent_scalar("tag:yaml.org,2002:str", stripped_data, style="|")


def prepare_message_for_yaml(message: dict, index: int) -> dict:
    """Prepare message data for YAML serialization

    Args:
        message: Raw message dict
        index: Message index (1-based)

    Returns:
        dict: Processed message dict
    """
    result = {"index": index, "role": message.get("role", "unknown")}

    content = message.get("content")
    if content:
        content_str = str(content)
        result["content"] = StripLiteralString(content_str)
    else:
        result["content"] = None

    # Handle tool_calls (assistant messages)
    tool_calls_in_msg = message.get("tool_calls")
    if tool_calls_in_msg:
        result["tool_calls"] = []
        for tc in tool_calls_in_msg:
            tc_dict = {"call_id": tc.get("id", ""), "function": {"name": tc.get("function", {}).get("name", ""), "arguments": {}}}
            try:
                args_str = tc.get("function", {}).get("arguments", "{}")
                tc_dict["function"]["arguments"] = json.loads(args_str)
            except json.JSONDecodeError:
                tc_dict["function"]["arguments"] = args_str

            result["tool_calls"].append(tc_dict)

    # Handle tool_call_id (tool messages)
    tool_call_id = message.get("tool_call_id")
    if tool_call_id:
        result["tool_call_id"] = tool_call_id

    return result


def _normalize_literal_text(value: str) -> str:
    """Remove trailing ASCII spaces per line to avoid blocking YAML literal block style"""
    if not value:
        return value

    lines = value.splitlines()
    normalized_lines = [line.rstrip(" ") for line in lines]
    normalized_text = "\n".join(normalized_lines)

    if value.endswith("\n"):
        normalized_text += "\n"

    return normalized_text


def _ensure_literal_block_style(value: Any) -> Any:
    """Recursively wrap strings containing newlines with StripLiteralString"""
    if isinstance(value, StripLiteralString):
        return StripLiteralString(_normalize_literal_text(str(value)))
    if isinstance(value, str) and "\n" in value:
        return StripLiteralString(_normalize_literal_text(value))
    if isinstance(value, dict):
        for key, nested_value in value.items():
            value[key] = _ensure_literal_block_style(nested_value)
        return value
    if isinstance(value, list):
        return [_ensure_literal_block_style(item) for item in value]
    return value


def build_yaml_structure(iteration: int, input_messages: list[dict], llm_response: Any, iteration_type: Optional[str] = None, debug_summary: Optional[dict] = None) -> dict:
    """Build YAML data structure

    Args:
        iteration: Iteration count (0-based, internally converted to 1-based)
        input_messages: Input message list
        llm_response: LLM response data (str or dict with content/tool_calls/finish_reason)
        iteration_type: Optional iteration type ("chunk_processing"|"compression"|"finalization")
        debug_summary: Optional debug info dict

    Returns:
        dict: YAML data structure
    """
    iteration_num = iteration + 1

    # === 1. Iteration Metadata ===
    iteration_meta = {"number": iteration_num, "timestamp": datetime.now(timezone.utc).isoformat()}
    if iteration_type:
        iteration_meta["type"] = iteration_type
    data = {"iteration": iteration_meta}

    # === 2. Input Messages ===
    data["input_messages"] = {"total_count": len(input_messages), "messages": [prepare_message_for_yaml(message=msg, index=i) for i, msg in enumerate(input_messages, 1)]}

    # === 3. LLM Output ===
    # Handle both str (old-style) and dict (new-style) llm_response
    if isinstance(llm_response, str):
        llm_response = {"content": llm_response, "tool_calls": None, "finish_reason": "stop"}

    finish_reason = llm_response.get("finish_reason", "stop")
    reasoning_content = llm_response.get("reasoning_content")
    content = llm_response.get("content")
    tool_calls = llm_response.get("tool_calls")

    data["llm_output"] = {"finish_reason": finish_reason, "reasoning_content": None, "content": None, "tool_calls": []}

    if reasoning_content:
        data["llm_output"]["reasoning_content"] = StripLiteralString(reasoning_content)

    if content:
        data["llm_output"]["content"] = StripLiteralString(content)

    if tool_calls:
        for tc in tool_calls:
            tc_dict = {"call_id": tc["id"], "type": tc.get("type", "function"), "function": {"name": tc["function"]["name"], "arguments": {}}}
            try:
                tc_dict["function"]["arguments"] = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                tc_dict["function"]["arguments"] = tc["function"]["arguments"]

            data["llm_output"]["tool_calls"].append(tc_dict)

    # === 4. Debug Summary ===
    data["debug_summary"] = debug_summary if debug_summary else {}

    return data


def save_iteration_log_as_yaml(
    log_file: Path,
    iteration: int,
    input_messages: list[dict],
    llm_response: Any,
    iteration_type: Optional[str] = None,
    debug_summary: Optional[dict] = None,
) -> None:
    """Save iteration log as YAML

    Args:
        log_file: Log file path (.yaml)
        iteration: Iteration count (0-based, internally converted to 1-based)
        input_messages: Input message list
        llm_response: LLM response (str or dict with content/tool_calls/finish_reason/reasoning_content)
        iteration_type: Optional iteration type ("chunk_processing"|"compression"|"finalization")
        debug_summary: Optional debug info dict
    """
    yaml_data = build_yaml_structure(iteration=iteration, input_messages=input_messages, llm_response=llm_response, iteration_type=iteration_type, debug_summary=debug_summary)

    yaml_data = _ensure_literal_block_style(yaml_data)

    class CustomDumper(yaml.Dumper):
        pass

    CustomDumper.add_representer(StripLiteralString, strip_literal_str_representer)

    with open(log_file, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, Dumper=CustomDumper, allow_unicode=True, default_flow_style=False, sort_keys=False, indent=2, width=120)
