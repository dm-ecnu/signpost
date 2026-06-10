"""Structured trace logging system (JSONL format)
TODO: Migrate to OpenTelemetry
TODO: docs/opentelemetry_migration_design.md

Core features:
- Single JSONL file per trace session
- Mandatory logging (no disable option)
- Thread-safe event serialization
- Hierarchical event structure (step > llm_call/tool_call)
- Complete content recording for replay and analysis

Core classes:
- TraceSession: Task-level session management (creates log directory, manages Logger)
- TraceEmitter: Agent-level event emitter (carries context: trace_id, agent_id, agent_type)
- TraceSpan: Context manager, auto-records start/end events
"""

import json
import logging
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from deepresearch.types import TraceStatus

logger = logging.getLogger(__name__)


# ===== JSONL output formatter =====


class JSONLFormatter(logging.Formatter):
    """Custom formatter that outputs JSON lines"""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON line

        record.msg should be a dict containing event data.
        """
        if isinstance(record.msg, dict):
            return json.dumps(record.msg, ensure_ascii=False, default=str)
        return super().format(record)


# ===== TraceSession: Task-level session management =====


class TraceSession:
    """Task-level trace session manager

    Responsibilities:
    1. Create log directory and initialize logging.Logger
    2. Generate trace_id and manage folder naming
    3. Maintain global event sequence number (thread-safe)
    4. Provide TraceEmitter creation factory method

    Error handling:
    - Initialization failures are silently caught (no exceptions thrown)
    - Failed sessions mark themselves as unavailable
    - All subsequent emit() calls become no-ops

    Usage:
        session = TraceSession(task="Research task", log_root=Path("logs"))
        emitter = session.create_emitter(agent_id="supervisor", agent_type="Supervisor")
        emitter.emit("step_start", step_id="xxx", step_index=1)
    """

    def __init__(self, task: str, log_root: Path = Path("logs"), trace_id: Optional[str] = None):
        self.task = task
        self.trace_id = trace_id or str(uuid.uuid4())
        self.trace_id_short = self.trace_id.split("-")[0]
        self.created_at = datetime.now(timezone.utc)

        self._sequence_counter = 0
        self._sequence_lock = threading.Lock()

        self._emitters: Dict[str, "TraceEmitter"] = {}
        self._emitters_lock = threading.Lock()

        self._available = False
        self._log_path: Optional[Path] = None
        self._logger: Optional[logging.Logger] = None

        try:
            self._initialize_logging(log_root)
            self._available = True
            logger.info(f"TraceSession initialized: {self._log_path}")
        except Exception as e:
            logger.warning(f"TraceSession initialization failed: {e}, logging disabled")
            self._available = False

    def _initialize_logging(self, log_root: Path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = log_root / f"{timestamp}_{self.trace_id_short}"
        log_dir.mkdir(parents=True, exist_ok=True)

        self._log_path = log_dir / "trace.jsonl"

        logger_name = f"trace_{self.trace_id_short}"
        self._logger = logging.getLogger(logger_name)
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False

        handler = logging.FileHandler(self._log_path, encoding="utf-8")
        handler.setFormatter(JSONLFormatter())
        self._logger.addHandler(handler)

    def _next_sequence(self) -> int:
        with self._sequence_lock:
            self._sequence_counter += 1
            return self._sequence_counter

    @property
    def log_dir(self) -> Optional[Path]:
        return self._log_path.parent if self._log_path else None

    @property
    def is_available(self) -> bool:
        return self._available

    def close(self):
        if self._logger:
            for handler in self._logger.handlers[:]:
                handler.close()
                self._logger.removeHandler(handler)
        self._available = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def create_emitter(self, agent_id: str, agent_type: str, **metadata) -> "TraceEmitter":
        emitter = TraceEmitter(
            session=self,
            agent_id=agent_id,
            agent_type=agent_type,
            **metadata,
        )

        with self._emitters_lock:
            self._emitters[agent_id] = emitter

        self.emit(
            event_type="agent_start",
            agent_id=agent_id,
            agent_type=agent_type,
            metadata=metadata if metadata else None,
        )

        return emitter

    def emit(self, event_type: str, agent_id: str, agent_type: str, **payload) -> Optional[str]:
        if not self._available:
            return None

        try:
            event_id = str(uuid.uuid4())
            sequence = self._next_sequence()
            timestamp = datetime.now(timezone.utc).isoformat()

            event = {
                "event_id": event_id,
                "event_type": event_type,
                "timestamp": timestamp,
                "trace_id": self.trace_id,
                "agent_id": agent_id,
                "agent_type": agent_type,
                "sequence": sequence,
                **payload,
            }

            self._logger.info(event)
            return event_id

        except Exception as e:
            logger.warning(f"Failed to emit event {event_type}: {e}")
            return None

    def emit_trace_start(self, config: Dict[str, Any]) -> Optional[str]:
        return self.emit(
            event_type="trace_start",
            agent_id="system",
            agent_type="System",
            task=self.task,
            config=config,
            log_path=str(self._log_path) if self._log_path else None,
        )

    def emit_final(self, status: TraceStatus, final_report: str, total_steps: int, total_tool_calls: int, total_llm_calls: int) -> Optional[str]:
        return self.emit(
            event_type="final",
            agent_id="system",
            agent_type="System",
            status=status.value,
            final_report=final_report,
            total_steps=total_steps,
            total_tool_calls=total_tool_calls,
            total_llm_calls=total_llm_calls,
        )

    def collect_statistics(self) -> Dict[str, Any]:
        stats = {
            "total_agents": len(self._emitters),
            "total_steps": 0,
            "total_llm_calls": 0,
            "total_tool_calls": 0,
            "total_compacts": 0,
            "by_agent": {},
        }

        with self._emitters_lock:
            for agent_id, emitter in self._emitters.items():
                agent_stats = emitter.get_statistics()
                stats["by_agent"][agent_id] = agent_stats
                stats["total_steps"] += agent_stats.get("steps", 0)
                stats["total_llm_calls"] += agent_stats.get("llm_calls", 0)
                stats["total_tool_calls"] += agent_stats.get("tool_calls", 0)
                stats["total_compacts"] += agent_stats.get("compacts", 0)

        return stats


# ===== TraceEmitter: Agent-level event emitter =====


class TraceEmitter:
    """Agent-level event emitter

    Carries context (trace_id, agent_id, agent_type) and provides:
    - emit(event_type, **payload): Direct event emission
    - span(event_type, **meta): Context manager for start/end events

    Usage:
        emitter = session.create_emitter("supervisor", "Supervisor")

        # Direct emission
        emitter.emit("agent_start", system_prompt="...", tools=[...])

        # Using span context manager
        with emitter.step_span(step_index=1) as step:
            with emitter.llm_span(model_id="gpt-4", messages=[...]) as llm:
                # ... LLM call ...
                llm.set_result(content="...", tool_calls=[...])
            with emitter.tool_span(tool_name="search", tool_call_id="tc_123") as tool:
                # ... Tool execution ...
                tool.set_result(output="...", status=TraceStatus.SUCCESS)
    """

    def __init__(self, session: TraceSession, agent_id: str, agent_type: str, **metadata):
        self._session = session
        self.agent_id = agent_id
        self.agent_type = agent_type
        self._metadata = metadata

        self._step_counter = 0
        self._llm_call_counter = 0
        self._tool_call_counter = 0
        self._compact_counter = 0
        self._total_tokens_used = 0
        self._counters_lock = threading.Lock()

        self._current_step_id: Optional[str] = None
        self._current_step_event_id: Optional[str] = None

    @property
    def trace_id(self) -> str:
        return self._session.trace_id

    @property
    def llm_call_counter(self) -> int:
        with self._counters_lock:
            return self._llm_call_counter

    @property
    def tool_call_counter(self) -> int:
        with self._counters_lock:
            return self._tool_call_counter

    @property
    def step_counter(self) -> int:
        with self._counters_lock:
            return self._step_counter

    def emit(self, event_type: str, **payload) -> Optional[str]:
        if event_type in ("tool_call_start", "tool_call_end") and "step_id" not in payload:
            if self._current_step_id:
                payload["step_id"] = self._current_step_id

        return self._session.emit(
            event_type=event_type,
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            **payload,
        )

    def set_current_step(self, step_id: Optional[str]) -> None:
        self._current_step_id = step_id

    def get_statistics(self) -> Dict[str, Any]:
        with self._counters_lock:
            return {
                "steps": self._step_counter,
                "llm_calls": self._llm_call_counter,
                "tool_calls": self._tool_call_counter,
                "compacts": self._compact_counter,
                "tokens_used": self._total_tokens_used,
            }

    # ===== Agent lifecycle events =====

    def emit_agent_start(self, system_prompt: str, tools: List[Dict[str, Any]], initialization: Optional[Dict] = None) -> Optional[str]:
        return self.emit(
            "agent_start",
            system_prompt=system_prompt,
            tools=tools,
            initialization=initialization or {"created_at": datetime.now(timezone.utc).isoformat()},
        )

    def emit_error(self, where: str, error_type: str, error_message: str, traceback: Optional[str] = None) -> Optional[str]:
        return self.emit(
            "error",
            where=where,
            error_type=error_type,
            error_message=error_message,
            traceback=traceback,
        )

    # ===== Span context managers =====

    @contextmanager
    def step_span(self, step_index: int) -> Iterator["StepSpan"]:
        with self._counters_lock:
            self._step_counter += 1
            step_id = f"{self.trace_id[:8]}.{self.agent_id}.step{step_index}"

        span = StepSpan(
            emitter=self,
            step_id=step_id,
            step_index=step_index,
        )

        self._current_step_id = step_id

        try:
            span._emit_start()
            self._current_step_event_id = span._start_event_id
            yield span
        except Exception as e:
            span._emit_error(e)
            raise
        finally:
            span._emit_end()
            self._current_step_id = None
            self._current_step_event_id = None

    @contextmanager
    def llm_span(
        self,
        model_id: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        stream: bool = True,
        step_id: Optional[str] = None,
    ) -> Iterator["LLMSpan"]:
        with self._counters_lock:
            self._llm_call_counter += 1

        effective_step_id = step_id if step_id is not None else self._current_step_id
        effective_parent_id = None if step_id is not None else self._current_step_event_id

        span = LLMSpan(
            emitter=self,
            model_id=model_id,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            stream=stream,
            step_id=effective_step_id,
            parent_event_id=effective_parent_id,
        )

        try:
            span._emit_start()
            yield span
        except Exception as e:
            span._emit_error(e)
            raise
        finally:
            span._emit_end()
            if span._usage:
                with self._counters_lock:
                    self._total_tokens_used += span._usage.get("total_tokens", 0)

    @contextmanager
    def tool_span(
        self,
        tool_name: str,
        tool_call_id: str,
        arguments_raw: str,
        arguments: Optional[Dict[str, Any]] = None,
        arguments_error: Optional[str] = None,
    ) -> Iterator["ToolSpan"]:
        with self._counters_lock:
            self._tool_call_counter += 1

        span = ToolSpan(
            emitter=self,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            arguments_raw=arguments_raw,
            arguments=arguments,
            arguments_error=arguments_error,
            step_id=self._current_step_id,
            parent_event_id=self._current_step_event_id,
        )

        try:
            span._emit_start()
            yield span
        except Exception as e:
            span._emit_error(e)
            raise
        finally:
            span._emit_end()

    # ===== Direct event methods (for non-span events) =====

    def emit_compact(
        self,
        before_compact: Dict[str, Any],
        after_compact: Dict[str, Any],
        deleted_chunks: List[Dict[str, Any]],
        chunk_scores: List[Dict[str, Any]],
        duration_ms: int,
    ) -> Optional[str]:
        with self._counters_lock:
            self._compact_counter += 1

        return self.emit(
            "compact",
            step_id=self._current_step_id,
            before_compact=before_compact,
            after_compact=after_compact,
            deleted_chunks=deleted_chunks,
            chunk_scores=chunk_scores,
            duration_ms=duration_ms,
        )

    # ===== Compatibility methods (for gradual migration) =====

    def write_config(self, config_data: Dict[str, Any]):
        self.emit_agent_start(
            system_prompt=config_data.get("system_prompt", ""),
            tools=config_data.get("tools", []),
            initialization=config_data.get("initialization"),
        )

    def write_metadata(self):
        pass

    def write_final_output(self, content: str):
        pass

    def write_compact(self, compact_data: Dict[str, Any]) -> str:
        timing = compact_data.get("timing", {})
        started_at = timing.get("triggered_at", "")
        completed_at = timing.get("completed_at", "")

        duration_ms = 0
        try:
            if started_at and completed_at:
                start_dt = datetime.fromisoformat(started_at.rstrip("Z"))
                end_dt = datetime.fromisoformat(completed_at.rstrip("Z"))
                duration_ms = int((end_dt - start_dt).total_seconds() * 1000)
        except Exception:
            duration_ms = int(timing.get("duration_seconds", 0) * 1000)

        event_id = self.emit_compact(
            before_compact=compact_data.get("before_compact", {}),
            after_compact=compact_data.get("after_compact", {}),
            deleted_chunks=compact_data.get("deleted_chunks", []),
            chunk_scores=compact_data.get("scoring", {}).get("chunk_scores", []),
            duration_ms=duration_ms,
        )
        return event_id or ""

    def log_llm_call_simple(
        self,
        step_number: int,
        messages: List[Any],
        openai_tools: List[Dict],
        response_content: Optional[str],
        tool_calls: Optional[List[Any]],
        start_time: datetime,
        end_time: datetime,
        model_id: str,
        max_tokens: int,
        usage: Optional[Dict[str, int]] = None,
        description: str = "react",
        stream: bool = True,
    ) -> str:
        try:
            formatted_messages = format_messages_for_log(messages)

            duration_ms = int((end_time - start_time).total_seconds() * 1000)

            start_event_id = self.emit(
                "llm_call_start",
                step_id=self._current_step_id,
                parent_event_id=self._current_step_event_id,
                model_id=model_id,
                messages=formatted_messages,
                tools=openai_tools,
                max_tokens=max_tokens,
                stream=stream,
            )

            tool_calls_data = [tc.to_dict() for tc in tool_calls] if tool_calls else None
            self.emit(
                "llm_call_end",
                step_id=self._current_step_id,
                parent_event_id=start_event_id,
                content=response_content,
                tool_calls=tool_calls_data,
                usage=usage,
                finish_reason="tool_calls" if tool_calls else "stop",
                duration_ms=duration_ms,
            )

            if usage:
                with self._counters_lock:
                    self._total_tokens_used += usage.get("total_tokens", 0)

            with self._counters_lock:
                self._llm_call_counter += 1

            return start_event_id or ""
        except Exception as e:
            logger.warning(f"Failed to log LLM call: {e}")
            return ""

    def log_tool_call_simple(
        self,
        tool_call: Any,
        tool_name: str,
        result: Optional[str],
        error: Optional[Exception],
        start_time: datetime,
        end_time: datetime,
    ) -> Optional[str]:
        try:
            import json as json_module

            arguments_raw = tool_call.function.arguments
            arguments = None
            arguments_error = None
            try:
                arguments = json_module.loads(arguments_raw) if arguments_raw else {}
            except json_module.JSONDecodeError as e:
                arguments_error = str(e)

            duration_ms = int((end_time - start_time).total_seconds() * 1000)

            start_event_id = self.emit(
                "tool_call_start",
                step_id=self._current_step_id,
                parent_event_id=self._current_step_event_id,
                tool_name=tool_name,
                tool_call_id=tool_call.id,
                arguments_raw=arguments_raw,
                arguments=arguments,
                arguments_error=arguments_error,
            )

            self.emit(
                "tool_call_end",
                step_id=self._current_step_id,
                parent_event_id=start_event_id,
                tool_name=tool_name,
                tool_call_id=tool_call.id,
                status=TraceStatus.ERROR.value if error else TraceStatus.SUCCESS.value,
                output=result if not error else f"ERROR: {str(error)}",
                error_type=type(error).__name__ if error else None,
                error_message=str(error) if error else None,
                duration_ms=duration_ms,
            )

            with self._counters_lock:
                self._tool_call_counter += 1

            return start_event_id
        except Exception as e:
            logger.warning(f"Failed to log tool call: {e}")
            return None

    def log_step_simple(
        self,
        step_number: int,
        start_time: datetime,
        end_time: datetime,
        llm_call_counter: int,
        iteration: int,
        response_type: str,
        tool_call_summaries: List[Dict],
        next_action: str,
    ) -> int:
        try:
            duration_ms = int((end_time - start_time).total_seconds() * 1000)
            step_id = f"{self.trace_id[:8]}.{self.agent_id}.step{step_number}"

            start_event_id = self.emit(
                "step_start",
                step_id=step_id,
                step_index=step_number,
                iteration=step_number,
            )

            llm_call_ids = [f"llm_{i}" for i in range(1, llm_call_counter + 1)] if llm_call_counter > 0 else []

            self.emit(
                "step_end",
                step_id=step_id,
                step_index=step_number,
                parent_event_id=start_event_id,
                response_type=response_type,
                next_action=next_action,
                llm_call_ids=llm_call_ids,
                tool_call_ids=[tc.get("tool_call_id") for tc in tool_call_summaries],
                duration_ms=duration_ms,
            )

            return step_number
        except Exception as e:
            logger.warning(f"Failed to log step: {e}")
            return 0


# ===== Span classes =====


class StepSpan:
    """Step event span"""

    def __init__(self, emitter: TraceEmitter, step_id: str, step_index: int):
        self._emitter = emitter
        self._step_id = step_id
        self._step_index = step_index
        self._start_time = datetime.now(timezone.utc)
        self._start_event_id: Optional[str] = None

        self._response_type: str = "empty"
        self._next_action: str = "terminate"
        self._llm_call_ids: List[str] = []
        self._tool_call_ids: List[str] = []
        self._thought: Optional[str] = None
        self._action: Optional[Dict[str, Any]] = None

    @property
    def step_id(self) -> str:
        return self._step_id

    @property
    def step_index(self) -> int:
        return self._step_index

    @property
    def emitter(self) -> TraceEmitter:
        return self._emitter

    def set_thought(self, thought: str):
        self._thought = thought

    def set_action(self, action: Dict[str, Any]):
        self._action = action

    def set_result(
        self,
        response_type: str = "empty",
        next_action: str = "terminate",
        llm_call_ids: Optional[List[str]] = None,
        tool_call_ids: Optional[List[str]] = None,
    ):
        self._response_type = response_type
        self._next_action = next_action
        if llm_call_ids:
            self._llm_call_ids = llm_call_ids
        if tool_call_ids:
            self._tool_call_ids = tool_call_ids

    def _emit_start(self):
        self._start_event_id = self._emitter.emit(
            "step_start",
            step_id=self._step_id,
            step_index=self._step_index,
            iteration=self._step_index,
        )

    def _emit_end(self):
        end_time = datetime.now(timezone.utc)
        duration_ms = int((end_time - self._start_time).total_seconds() * 1000)

        self._emitter.emit(
            "step_end",
            step_id=self._step_id,
            step_index=self._step_index,
            parent_event_id=self._start_event_id,
            response_type=self._response_type,
            next_action=self._next_action,
            thought=self._thought,
            action=self._action,
            llm_call_ids=self._llm_call_ids,
            tool_call_ids=self._tool_call_ids,
            duration_ms=duration_ms,
        )

    def _emit_error(self, error: Exception):
        self._emitter.emit_error(
            where=f"step_{self._step_index}",
            error_type=type(error).__name__,
            error_message=str(error),
        )


class LLMSpan:
    """LLM call event span"""

    def __init__(
        self,
        emitter: TraceEmitter,
        model_id: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        max_tokens: Optional[int],
        stream: bool,
        step_id: Optional[str],
        parent_event_id: Optional[str],
    ):
        self._emitter = emitter
        self._model_id = model_id
        self._messages = messages
        self._tools = tools
        self._max_tokens = max_tokens
        self._stream = stream
        self._step_id = step_id
        self._parent_event_id = parent_event_id
        self._start_time = datetime.now(timezone.utc)
        self._start_event_id: Optional[str] = None

        self._content: Optional[str] = None
        self._tool_calls: Optional[List[Dict[str, Any]]] = None
        self._usage: Optional[Dict[str, int]] = None
        self._finish_reason: str = "stop"

    def set_result(
        self,
        content: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        usage: Optional[Dict[str, int]] = None,
        finish_reason: str = "stop",
    ):
        self._content = content
        self._tool_calls = tool_calls
        self._usage = usage
        self._finish_reason = finish_reason

    def _emit_start(self):
        self._start_event_id = self._emitter.emit(
            "llm_call_start",
            step_id=self._step_id,
            parent_event_id=self._parent_event_id,
            model_id=self._model_id,
            messages=self._messages,
            tools=self._tools,
            max_tokens=self._max_tokens,
            stream=self._stream,
        )

    def _emit_end(self):
        end_time = datetime.now(timezone.utc)
        duration_ms = int((end_time - self._start_time).total_seconds() * 1000)

        self._emitter.emit(
            "llm_call_end",
            step_id=self._step_id,
            parent_event_id=self._start_event_id,
            content=self._content,
            tool_calls=self._tool_calls,
            usage=self._usage,
            finish_reason=self._finish_reason,
            duration_ms=duration_ms,
        )

    def _emit_error(self, error: Exception):
        self._emitter.emit_error(
            where="llm_call",
            error_type=type(error).__name__,
            error_message=str(error),
        )


class ToolSpan:
    """Tool call event span"""

    def __init__(
        self,
        emitter: TraceEmitter,
        tool_name: str,
        tool_call_id: str,
        arguments_raw: str,
        arguments: Optional[Dict[str, Any]],
        arguments_error: Optional[str],
        step_id: Optional[str],
        parent_event_id: Optional[str],
    ):
        self._emitter = emitter
        self._tool_name = tool_name
        self._tool_call_id = tool_call_id
        self._arguments_raw = arguments_raw
        self._arguments = arguments
        self._arguments_error = arguments_error
        self._step_id = step_id
        self._parent_event_id = parent_event_id
        self._start_time = datetime.now(timezone.utc)
        self._start_event_id: Optional[str] = None

        self._status: str = TraceStatus.SUCCESS.value
        self._output: Optional[str] = None
        self._error_type: Optional[str] = None
        self._error_message: Optional[str] = None

    def set_result(
        self,
        status: TraceStatus = TraceStatus.SUCCESS,
        output: Optional[str] = None,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
    ):
        self._status = status.value
        self._output = output
        self._error_type = error_type
        self._error_message = error_message

    def _emit_start(self):
        self._start_event_id = self._emitter.emit(
            "tool_call_start",
            step_id=self._step_id,
            parent_event_id=self._parent_event_id,
            tool_name=self._tool_name,
            tool_call_id=self._tool_call_id,
            arguments_raw=self._arguments_raw,
            arguments=self._arguments,
            arguments_error=self._arguments_error,
        )

    def _emit_end(self):
        end_time = datetime.now(timezone.utc)
        duration_ms = int((end_time - self._start_time).total_seconds() * 1000)

        self._emitter.emit(
            "tool_call_end",
            step_id=self._step_id,
            parent_event_id=self._start_event_id,
            tool_name=self._tool_name,
            tool_call_id=self._tool_call_id,
            status=self._status,
            output=self._output,
            error_type=self._error_type,
            error_message=self._error_message,
            duration_ms=duration_ms,
        )

    def _emit_error(self, error: Exception):
        self._status = TraceStatus.ERROR.value
        self._error_type = type(error).__name__
        self._error_message = str(error)


# ===== Helper functions =====


def format_messages_for_log(messages: List[Any], system_prompt_placeholder: str = "<see agent_start event>") -> List[Dict[str, Any]]:
    formatted = []
    for msg in messages:
        if hasattr(msg, "to_openai_dict"):
            msg_dict = msg.to_openai_dict()
        elif isinstance(msg, dict):
            msg_dict = msg
        else:
            msg_dict = {"role": "unknown", "content": str(msg)}

        if msg_dict.get("role") == "system":
            msg_dict = msg_dict.copy()
            msg_dict["content"] = system_prompt_placeholder

        formatted.append(msg_dict)

    return formatted


# ===== Backward compatibility aliases =====

TraceLogger = TraceSession
AgentLogger = TraceEmitter


def create_trace_logger(task: str, log_root: Path = Path("logs"), trace_id: Optional[str] = None) -> TraceSession:
    return TraceSession(task=task, log_root=log_root, trace_id=trace_id)
