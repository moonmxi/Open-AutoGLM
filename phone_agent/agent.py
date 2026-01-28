"""Main PhoneAgent class for orchestrating phone automation."""

import base64
import json
import math
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable

import io
from collections import deque

from phone_agent.actions import ActionHandler
from phone_agent.actions.handler import finish, parse_action
from phone_agent.device_factory import get_device_factory
from phone_agent.model import ModelClient, ModelConfig
from phone_agent.model.client import MessageBuilder
from phone_agent.workflow.prompt_rules import ACTION_FORMAT_RETRY_CONTENT


def _ui_messages(lang: str = "cn") -> dict[str, str]:
    # Leak-only workflows shouldn't require the full phone_agent.config.* stack.
    if (lang or "").lower().startswith("en"):
        return {
            "thinking": "Thinking",
            "action": "Action",
            "task_completed": "Task Completed",
            "done": "Done",
        }
    return {
        "thinking": "思考过程",
        "action": "执行动作",
        "task_completed": "任务完成",
        "done": "完成",
    }


@dataclass
class AgentConfig:
    """Configuration for the PhoneAgent."""

    max_steps: int = 100
    device_id: str | None = None
    lang: str = "cn"
    system_prompt: str | None = None
    verbose: bool = True
    screen_info_hook: Callable[[str, str], dict[str, Any]] | None = None
    same_screen_threshold: float = 0.98
    max_same_screen_steps: int = 10
    debug_prompt: bool = True

    def __post_init__(self):
        if self.system_prompt is None:
            # Leak mode sets system_prompt explicitly in main.py; keep a minimal fallback here.
            self.system_prompt = (
                "You are a HarmonyOS UI automation agent.\n"
                "Strictly output:\n<think>{think}</think>\n<answer>{action}</answer>\n"
                "Allowed actions: do(...) or finish(...).\n"
            )


@dataclass
class StepResult:
    """Result of a single agent step."""

    success: bool
    finished: bool
    action: dict[str, Any] | None
    thinking: str
    message: str | None = None


class PhoneAgent:
    """
    AI-powered agent for automating HarmonyOS phone interactions.

    The agent uses a vision-language model to understand screen content
    and decide on actions to complete user tasks.

    Args:
        model_config: Configuration for the AI model.
        agent_config: Configuration for the agent behavior.
        confirmation_callback: Optional callback for sensitive action confirmation.
        takeover_callback: Optional callback for takeover requests.

    Example:
        >>> from phone_agent import PhoneAgent
        >>> from phone_agent.model import ModelConfig
        >>>
        >>> model_config = ModelConfig(base_url="http://localhost:8000/v1")
        >>> agent = PhoneAgent(model_config)
        >>> agent.run("Open WeChat and send a message to John")
    """

    def __init__(
        self,
        model_config: ModelConfig | None = None,
        agent_config: AgentConfig | None = None,
        confirmation_callback: Callable[[str], bool] | None = None,
        takeover_callback: Callable[[str], None] | None = None,
    ):
        self.model_config = model_config or ModelConfig()
        self.agent_config = agent_config or AgentConfig()

        self.model_client = ModelClient(self.model_config)
        self.action_handler = ActionHandler(
            device_id=self.agent_config.device_id,
            confirmation_callback=confirmation_callback,
            takeover_callback=takeover_callback,
        )

        self._context: list[dict[str, Any]] = []
        self._step_count = 0
        self._prev_screen_hash: int | None = None
        self._same_screen_streak = 0
        self._recent_screen_hashes: deque[int] = deque(maxlen=12)
        self._last_action_signature: str | None = None
        self._repeat_action_streak = 0
        self._last_action_result: dict[str, Any] | None = None
        self._reference_images_stripped = False
        self._leak_phase: str | None = None
        self._last_action_before_base64: str | None = None
        self._last_action_after_base64: str | None = None

    def run(self, task: str, *, extra_messages: list[dict[str, Any]] | None = None) -> str:
        """
        Run the agent to complete a task.

        Args:
            task: Natural language description of the task.
            extra_messages: Optional messages to inject before the first live observation.

        Returns:
            Final message from the agent.
        """
        self._context = []
        self._step_count = 0
        self._prev_screen_hash = None
        self._same_screen_streak = 0
        self._recent_screen_hashes.clear()
        self._last_action_signature = None
        self._repeat_action_streak = 0
        self._last_action_result = None
        self._reference_images_stripped = False
        self._leak_phase = None
        self._last_action_before_base64 = None
        self._last_action_after_base64 = None
        self._leak_phase = None

        # First step with user prompt
        result = self._execute_step(task, is_first=True, extra_messages=extra_messages)

        if result.finished:
            return result.message or "Task completed"

        # Continue until finished or max steps reached
        while self._step_count < self.agent_config.max_steps:
            result = self._execute_step(is_first=False)

            if result.finished:
                return result.message or "Task completed"

        return "Max steps reached"

    def step(
        self,
        task: str | None = None,
        *,
        extra_messages: list[dict[str, Any]] | None = None,
    ) -> StepResult:
        """
        Execute a single step of the agent.

        Useful for manual control or debugging.

        Args:
            task: Task description (only needed for first step).

        Returns:
            StepResult with step details.
        """
        is_first = len(self._context) == 0

        if is_first and not task:
            raise ValueError("Task is required for the first step")

        return self._execute_step(task, is_first, extra_messages=extra_messages)

    def reset(self) -> None:
        """Reset the agent state for a new task."""
        self._context = []
        self._step_count = 0
        self._prev_screen_hash = None
        self._same_screen_streak = 0
        self._recent_screen_hashes.clear()
        self._last_action_signature = None
        self._repeat_action_streak = 0
        self._last_action_result = None
        self._reference_images_stripped = False

    @staticmethod
    def _dhash_from_base64(image_base64: str, hash_size: int = 8) -> int | None:
        try:
            from PIL import Image

            raw = base64.b64decode(image_base64)
            img = Image.open(io.BytesIO(raw)).convert("L")
            img = img.resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
            pixels = list(img.getdata())
            result = 0
            for row in range(hash_size):
                row_start = row * (hash_size + 1)
                for col in range(hash_size):
                    left = pixels[row_start + col]
                    right = pixels[row_start + col + 1]
                    bit = 1 if left > right else 0
                    result = (result << 1) | bit
            return result
        except Exception:
            return None

    @staticmethod
    def _hash_similarity(a: int, b: int, bits: int = 64) -> float:
        dist = (a ^ b).bit_count()
        return 1.0 - (dist / bits)

    def _execute_step(
        self,
        user_prompt: str | None = None,
        is_first: bool = False,
        extra_messages: list[dict[str, Any]] | None = None,
    ) -> StepResult:
        """Execute a single step of the agent loop."""
        self._step_count += 1

        # Capture current screen state
        device_factory = get_device_factory()
        screenshot = device_factory.get_screenshot(self.agent_config.device_id)
        current_app = device_factory.get_current_app(self.agent_config.device_id)

        # Simple loop guard: detect repeated same-screen + repeated action patterns.
        current_hash = None
        # Treat sensitive/black-like frames as unusable for hash-based loop detection.
        if not getattr(screenshot, "is_sensitive", False):
            current_hash = self._dhash_from_base64(screenshot.base64_data)
        screen_similarity_to_prev: float | None = None
        if current_hash is not None and self._prev_screen_hash is not None:
            screen_similarity_to_prev = self._hash_similarity(
                current_hash, self._prev_screen_hash
            )
            if screen_similarity_to_prev >= self.agent_config.same_screen_threshold:
                self._same_screen_streak += 1
            else:
                self._same_screen_streak = 0
        else:
            self._same_screen_streak = 0

        self._prev_screen_hash = current_hash
        if current_hash is not None:
            self._recent_screen_hashes.append(current_hash)

        def _is_two_state_cycle() -> bool:
            if len(self._recent_screen_hashes) < 6:
                return False
            a = self._recent_screen_hashes[-1]
            b = self._recent_screen_hashes[-2]
            if a == b:
                return False
            tail = list(self._recent_screen_hashes)[-6:]
            return tail == [a, b, a, b, a, b] or tail == [b, a, b, a, b, a]

        def _is_three_state_cycle() -> bool:
            if len(self._recent_screen_hashes) < 9:
                return False
            tail = list(self._recent_screen_hashes)[-9:]
            a, b, c = tail[-3], tail[-2], tail[-1]
            if len({a, b, c}) != 3:
                return False
            return tail == [a, b, c, a, b, c, a, b, c]

        if (
            (self._same_screen_streak >= self.agent_config.max_same_screen_steps and self._repeat_action_streak >= 2)
            or _is_two_state_cycle()
            or _is_three_state_cycle()
        ):
            message = (
                "Loop detected: the agent is not making progress (same-screen repeat or screen cycle). "
                "Likely causes: target screen mismatch, imprecise coordinates, or navigation bouncing. "
                "Try: lower reliance on coordinates (use UI cues), add a deterministic reset (Home->Launch), or reduce the action window."
            )
            return StepResult(
                success=False,
                finished=True,
                action=finish(message=message),
                thinking="",
                message=message,
            )

        extra_screen_info: dict[str, Any] = {}
        if self.agent_config.screen_info_hook is not None:
            try:
                extra_screen_info = self.agent_config.screen_info_hook(
                    current_app, screenshot.base64_data
                ) or {}
            except Exception:
                extra_screen_info = {}

        # Track leak-mode阶段，指导模型先对齐场景再复现动作。
        leak_mode_flag = bool(extra_screen_info.get("leak_mode"))
        if leak_mode_flag:
            # 移除所有推测性的分数和阶段提示，只保留最基础的状态标记
            extra_screen_info = {
                "leak_mode": True,
                "target_app": extra_screen_info.get("target_app"),
            }

        if self._last_action_result is not None:
            # If the last action produced no visible change, hint the model to adjust.
            if (
                screen_similarity_to_prev is not None
                and self._last_action_result.get("action", {}).get("_metadata") == "do"
                and self._last_action_result.get("action", {}).get("action") not in ("Wait",)
                and screen_similarity_to_prev >= self.agent_config.same_screen_threshold
            ):
                self._last_action_result = {
                    **self._last_action_result,
                    "success": False,
                    "message": "Last action produced no visible change (same screen).",
                }

            extra_screen_info = {
                **extra_screen_info,
                "last_action": self._last_action_result.get("action"),
                "last_action_ok": self._last_action_result.get("success"),
                "last_action_message": self._last_action_result.get("message"),
            }

        # 仅保留物理事实和状态，移除推测性分数
        extra_screen_info = {
            **extra_screen_info,
            "same_screen_streak": self._same_screen_streak,
            "screenshot_is_sensitive": getattr(screenshot, "is_sensitive", False),
        }
        if self._same_screen_streak >= 3:
            extra_screen_info["progress_hint"] = "Same screen repeated; change strategy (reset/back/home/launch or try different path)."
        if (
            self._same_screen_streak >= 2
            and self._repeat_action_streak >= 2
            and self._last_action_result is not None
            and not self._last_action_result.get("success", True)
        ):
            extra_screen_info["progress_hint"] = (
                "Repeated identical action with no progress; adjust target/coords or choose another navigation path (e.g., Back -> Home -> relaunch)."
            )

        # Build messages
        if is_first:
            self._context.append(
                MessageBuilder.create_system_message(self.agent_config.system_prompt)
            )

            if extra_messages:
                self._context.extend(extra_messages)

        screen_info = MessageBuilder.build_screen_info(current_app, **extra_screen_info)
        if is_first and user_prompt:
            text_content = f"{user_prompt}\n\n{screen_info}"
        else:
            text_content = f"** Screen Info **\n\n{screen_info}"

        # Build a multi-modal user message that can include previous step screenshots.
        parts: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{screenshot.base64_data}"}},
            {"type": "text", "text": text_content},
        ]
        if self._last_action_before_base64 and self._last_action_after_base64:
            parts.extend(
                [
                    {"type": "text", "text": "【上一步操作前截图】用于对比定位点击对象"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self._last_action_before_base64}"}},
                    {"type": "text", "text": "【上一步操作后截图】"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self._last_action_after_base64}"}},
                ]
            )

        self._context.append(MessageBuilder.create_user_message_from_parts(parts))

        if self.agent_config.debug_prompt:
            print("\n" + "=" * 50)
            print("【DEBUG PROMPT START】")
            print("-" * 30)
            
            # 策略：
            # 1. 总是打印 System Prompt (Index 0)
            # 2. 如果是第一轮对话（is_first），打印 Initial User Prompt (Index 1)
            # 3. 总是打印 Last Message (最新的一条)
            
            indices_to_print = {0, len(self._context) - 1}
            if is_first and len(self._context) > 1:
                indices_to_print.add(1)
            
            # 按顺序打印选中的消息
            for idx in sorted(indices_to_print):
                msg = self._context[idx]
                role = msg["role"].upper()
                print(f"[{role} MESSAGE (#{idx})]:")
                
                content = msg.get("content", "")
                if isinstance(content, list):
                    for i, part in enumerate(content):
                        p_type = part.get("type", "unknown")
                        if p_type == "text":
                            text_preview = part.get("text", "")
                            indented_text = "\n    ".join(text_preview.splitlines())
                            print(f"  Part {i} [Text]:\n    {indented_text}")
                        elif p_type == "image_url":
                            url = part.get("image_url", {}).get("url", "")
                            if url.startswith("data:"):
                                meta, _ = url.split(",", 1) if "," in url else (url, "")
                                print(f"  Part {i} [Image]: {meta},... (len={len(url)})")
                            else:
                                print(f"  Part {i} [Image]: {url}")
                else:
                    print(f"  [Content]: {content}")
                print("-" * 30)

            if len(self._context) > 3 and not is_first:
                 print(f"... (Skipped {len(self._context) - 3} intermediate messages) ...")
                 print("-" * 30)

            print("【DEBUG PROMPT END】")
            print("=" * 50 + "\n")

        # Get model response
        try:
            msgs = _ui_messages(self.agent_config.lang)
            print("\n" + "=" * 50)
            print(f"{msgs['thinking']}:")
            print("-" * 50)
            response = self.model_client.request(self._context)
        except Exception as e:
            if self.agent_config.verbose:
                traceback.print_exc()
            return StepResult(
                success=False,
                finished=True,
                action=None,
                thinking="",
                message=f"Model error: {e}",
            )

        # Parse action from response
        try:
            action = parse_action(response.action)
        except ValueError as e:
            # Fallback: try raw_content (e.g., malformed <answer><answer> or empty action field)
            fallback_parsed = None
            raw_resp = getattr(response, "raw_content", None)
            if raw_resp:
                try:
                    fallback_parsed = parse_action(raw_resp)
                except Exception:
                    fallback_parsed = None
            if fallback_parsed:
                action = fallback_parsed
            else:
                retry_parsed = None
                retry_response = None
                try:
                    self._context.append(MessageBuilder.create_user_message(ACTION_FORMAT_RETRY_CONTENT))
                    retry_response = self.model_client.request(self._context)
                    try:
                        retry_parsed = parse_action(retry_response.action)
                    except Exception:
                        raw_retry = getattr(retry_response, "raw_content", None)
                        if raw_retry:
                            retry_parsed = parse_action(raw_retry)
                except Exception:
                    retry_parsed = None

                if retry_parsed and retry_response is not None:
                    response = retry_response
                    action = retry_parsed
                else:
                    if self.agent_config.verbose:
                        traceback.print_exc()
                    action = finish(message=str(e))

        action_signature = json.dumps(action, sort_keys=True, ensure_ascii=False)
        if action_signature == self._last_action_signature:
            self._repeat_action_streak += 1
        else:
            self._repeat_action_streak = 0
        self._last_action_signature = action_signature

        if self.agent_config.verbose:
            # Print thinking process
            print("-" * 50)
            print(f"{msgs['action']}:")
            print(json.dumps(action, ensure_ascii=False, indent=2))
            print("=" * 50 + "\n")

        # Remove the live screen image from context to save space.
        # Keep any injected reference images (e.g., leak case pre/post screenshots) for anchoring.
        self._context[-1] = MessageBuilder.remove_images_from_message(self._context[-1])

        # Execute action
        try:
            result = self.action_handler.execute(
                action, screenshot.width, screenshot.height
            )
        except Exception as e:
            if self.agent_config.verbose:
                traceback.print_exc()
            result = self.action_handler.execute(
                finish(message=str(e)), screenshot.width, screenshot.height
            )

        # Capture post-action screenshot for next-step comparison (helps the model infer click effects).
        try:
            post_action_shot = device_factory.get_screenshot(self.agent_config.device_id)
            self._last_action_before_base64 = screenshot.base64_data
            self._last_action_after_base64 = post_action_shot.base64_data
        except Exception:
            self._last_action_before_base64 = None
            self._last_action_after_base64 = None

        # Keep reference screenshots in context for the whole run (scene alignment is the core of leak mode).

        self._last_action_result = {
            "action": action,
            "success": result.success,
            "message": result.message,
            "should_finish": result.should_finish,
        }

        # Add assistant response to context
        self._context.append(
            MessageBuilder.create_assistant_message(
                f"<think>{response.thinking}</think><answer>{response.action}</answer>"
            )
        )

        # Check if finished
        finished = action.get("_metadata") == "finish" or result.should_finish

        if finished and self.agent_config.verbose:
            msgs = _ui_messages(self.agent_config.lang)
            print("\n" + "=" * 50)
            print(
                f"{msgs['task_completed']}: {result.message or action.get('message', msgs['done'])}"
            )
            print("=" * 50 + "\n")

        return StepResult(
            success=result.success,
            finished=finished,
            action=action,
            thinking=response.thinking,
            message=result.message or action.get("message"),
        )

    @property
    def context(self) -> list[dict[str, Any]]:
        """Get the current conversation context."""
        return self._context.copy()

    @property
    def step_count(self) -> int:
        """Get the current step count."""
        return self._step_count

    def execute_do_sequence(
        self,
        do_steps: list[str],
        *,
        repeat_count: int = 1,
        inter_step_wait_s: float = 0.5,
    ) -> list[str]:
        """
        Execute a list of `do(...)` steps (as strings) on the connected device.

        This is intended for workflow controllers that want to replay a verified reproduction sequence
        without invoking the model.

        Returns:
            A list of per-run status messages.
        """
        if repeat_count < 1:
            raise ValueError("repeat_count must be >= 1")
        if not do_steps:
            raise ValueError("do_steps must not be empty")

        device_factory = get_device_factory()
        run_messages: list[str] = []

        for run_idx in range(repeat_count):
            for step_idx, step in enumerate(do_steps):
                action = parse_action(step)
                screenshot = device_factory.get_screenshot(self.agent_config.device_id)
                result = self.action_handler.execute(
                    action, screenshot.width, screenshot.height
                )
                if result.should_finish:
                    run_messages.append(
                        f"run={run_idx} finished early at step={step_idx}: {result.message or ''}"
                    )
                    break
                if not result.success:
                    run_messages.append(
                        f"run={run_idx} failed at step={step_idx}: {result.message or ''}"
                    )
                    break
                if inter_step_wait_s > 0:
                    time.sleep(inter_step_wait_s)
            else:
                run_messages.append(f"run={run_idx} ok")

        return run_messages

    def execute_repro_sequence(
        self,
        repro_sequence: dict[str, Any],
        *,
        repeat_count: int = 1,
        inter_step_wait_s: float = 0.5,
    ) -> list[str]:
        """
        Execute a `repro_sequence.json` payload produced by the agent finish(message=...).

        Expected minimal schema:
            { "steps": [ { "action": "Back" }, { "action": "Tap", "element": [x,y] }, ... ] }
        Backward compatible:
            { "steps": [ { "do": "do(action=...)" }, ... ] }
        """
        steps = repro_sequence.get("steps")
        if not isinstance(steps, list):
            raise ValueError("repro_sequence.steps must be a list")

        # New schema: structured action dicts
        if steps and isinstance(steps[0], dict) and "action" in steps[0]:
            device_factory = get_device_factory()
            run_messages: list[str] = []

            for run_idx in range(repeat_count):
                for step_idx, item in enumerate(steps):
                    if not isinstance(item, dict) or "action" not in item:
                        raise ValueError(
                            f"repro_sequence.steps[{step_idx}] must be an object with an 'action' field"
                        )
                    action = {"_metadata": "do", **item}
                    screenshot = device_factory.get_screenshot(self.agent_config.device_id)
                    result = self.action_handler.execute(
                        action, screenshot.width, screenshot.height
                    )
                    if result.should_finish:
                        run_messages.append(
                            f"run={run_idx} finished early at step={step_idx}: {result.message or ''}"
                        )
                        break
                    if not result.success:
                        run_messages.append(
                            f"run={run_idx} failed at step={step_idx}: {result.message or ''}"
                        )
                        break
                    if inter_step_wait_s > 0:
                        time.sleep(inter_step_wait_s)
                else:
                    run_messages.append(f"run={run_idx} ok")

            return run_messages

        # Old schema: do(...) strings
        do_steps: list[str] = []
        for i, item in enumerate(steps):
            if not isinstance(item, dict) or "do" not in item:
                raise ValueError(
                    f"repro_sequence.steps[{i}] must be an object with an 'action' or 'do' field"
                )
            do_steps.append(str(item["do"]))
        return self.execute_do_sequence(do_steps, repeat_count=repeat_count, inter_step_wait_s=inter_step_wait_s)
