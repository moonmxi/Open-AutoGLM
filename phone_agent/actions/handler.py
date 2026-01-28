"""Action handler for processing AI model outputs."""

import ast
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Callable

from phone_agent.config.timing import TIMING_CONFIG
from phone_agent.device_factory import get_device_factory


@dataclass
class ActionResult:
    """Result of an action execution."""

    success: bool
    should_finish: bool
    message: str | None = None
    requires_confirmation: bool = False


class ActionHandler:
    """
    Handles execution of actions from AI model output.

    Args:
        device_id: Optional ADB device ID for multi-device setups.
        confirmation_callback: Optional callback for sensitive action confirmation.
            Should return True to proceed, False to cancel.
        takeover_callback: Optional callback for takeover requests (login, captcha).
    """

    def __init__(
        self,
        device_id: str | None = None,
        confirmation_callback: Callable[[str], bool] | None = None,
        takeover_callback: Callable[[str], None] | None = None,
    ):
        self.device_id = device_id
        self.confirmation_callback = confirmation_callback or self._default_confirmation
        self.takeover_callback = takeover_callback or self._default_takeover

    def execute(
        self, action: dict[str, Any], screen_width: int, screen_height: int
    ) -> ActionResult:
        """
        Execute an action from the AI model.

        Args:
            action: The action dictionary from the model.
            screen_width: Current screen width in pixels.
            screen_height: Current screen height in pixels.

        Returns:
            ActionResult indicating success and whether to finish.
        """
        action_type = action.get("_metadata")

        if action_type == "finish":
            return ActionResult(
                success=True, should_finish=True, message=action.get("message")
            )

        if action_type != "do":
            return ActionResult(
                success=False,
                should_finish=True,
                message=f"Unknown action type: {action_type}",
            )

        action_name = action.get("action")
        handler_method = self._get_handler(action_name)

        if handler_method is None:
            return ActionResult(
                success=False,
                should_finish=False,
                message=f"Unknown action: {action_name}",
            )

        # Prefer physical display size (wm size) to map relative coords accurately.
        try:
            device_factory = get_device_factory()
            display_size = device_factory.get_display_size(self.device_id)
        except Exception:
            display_size = None
        display_width, display_height = (
            display_size if display_size else (screen_width, screen_height)
        )

        try:
            return handler_method(action, screen_width, screen_height, display_width, display_height)
        except Exception as e:
            return ActionResult(
                success=False, should_finish=False, message=f"Action failed: {e}"
            )

    def _get_handler(self, action_name: str) -> Callable | None:
        """Get the handler method for an action."""
        handlers = {
            "Launch": self._handle_launch,
            "Tap": self._handle_tap,
            "Type": self._handle_type,
            "Type_Name": self._handle_type,
            "Swipe": self._handle_swipe,
            "Back": self._handle_back,
            "Home": self._handle_home,
            "Double Tap": self._handle_double_tap,
            "Long Press": self._handle_long_press,
            "Wait": self._handle_wait,
            "Take_over": self._handle_takeover,
            "Note": self._handle_note,
            "Call_API": self._handle_call_api,
            "Interact": self._handle_interact,
        }
        return handlers.get(action_name)

    def _convert_relative_to_absolute(
        self, element: list[int], display_width: int, display_height: int
    ) -> tuple[int, int]:
        """Convert relative coordinates (0-1000) to absolute pixels."""
        x = int(element[0] / 1000 * display_width)
        y = int(element[1] / 1000 * display_height)
        return x, y

    def _handle_launch(self, action: dict, width: int, height: int, display_width: int, display_height: int) -> ActionResult:
        """Handle app launch action."""
        app_name = action.get("app")
        if not app_name:
            return ActionResult(False, False, "No app name specified")

        device_factory = get_device_factory()
        success = device_factory.launch_app(app_name, self.device_id)
        if success:
            return ActionResult(True, False)
        return ActionResult(False, False, f"App not found: {app_name}")

    def _handle_tap(self, action: dict, width: int, height: int, display_width: int, display_height: int) -> ActionResult:
        """Handle tap action."""
        element = action.get("element")
        if not element:
            return ActionResult(False, False, "No element coordinates")

        x, y = self._convert_relative_to_absolute(element, display_width, display_height)

        # Check for sensitive operation
        if "message" in action:
            if not self.confirmation_callback(action["message"]):
                return ActionResult(
                    success=False,
                    should_finish=True,
                    message="User cancelled sensitive operation",
                )

        device_factory = get_device_factory()
        device_factory.tap(x, y, self.device_id)
        return ActionResult(True, False)

    def _handle_type(self, action: dict, width: int, height: int, display_width: int, display_height: int) -> ActionResult:
        """Handle text input action."""
        text = action.get("text", "")

        device_factory = get_device_factory()

        # Record current keyboard so it can be restored after typing
        original_ime = device_factory.detect_and_set_keyboard(self.device_id)
        time.sleep(TIMING_CONFIG.action.keyboard_switch_delay)

        # Clear existing text and type new text
        device_factory.clear_text(self.device_id)
        time.sleep(TIMING_CONFIG.action.text_clear_delay)

        # Handle multiline text by splitting on newlines
        device_factory.type_text(text, self.device_id)
        time.sleep(TIMING_CONFIG.action.text_input_delay)

        # Restore original keyboard
        device_factory.restore_keyboard(original_ime, self.device_id)
        time.sleep(TIMING_CONFIG.action.keyboard_restore_delay)

        return ActionResult(True, False)

    def _handle_swipe(self, action: dict, width: int, height: int, display_width: int, display_height: int) -> ActionResult:
        """Handle swipe action."""
        start = action.get("start")
        end = action.get("end")

        if not start or not end:
            return ActionResult(False, False, "Missing swipe coordinates")

        start_x, start_y = self._convert_relative_to_absolute(start, display_width, display_height)
        end_x, end_y = self._convert_relative_to_absolute(end, display_width, display_height)

        device_factory = get_device_factory()
        device_factory.swipe(start_x, start_y, end_x, end_y, device_id=self.device_id)
        return ActionResult(True, False)

    def _handle_back(self, action: dict, width: int, height: int, display_width: int, display_height: int) -> ActionResult:
        """Handle back button action."""
        device_factory = get_device_factory()
        device_factory.back(self.device_id)
        return ActionResult(True, False)

    def _handle_home(self, action: dict, width: int, height: int, display_width: int, display_height: int) -> ActionResult:
        """Handle home button action."""
        device_factory = get_device_factory()
        device_factory.home(self.device_id)
        return ActionResult(True, False)

    def _handle_double_tap(self, action: dict, width: int, height: int, display_width: int, display_height: int) -> ActionResult:
        """Handle double tap action."""
        element = action.get("element")
        if not element:
            return ActionResult(False, False, "No element coordinates")

        x, y = self._convert_relative_to_absolute(element, display_width, display_height)
        device_factory = get_device_factory()
        device_factory.double_tap(x, y, self.device_id)
        return ActionResult(True, False)

    def _handle_long_press(self, action: dict, width: int, height: int, display_width: int, display_height: int) -> ActionResult:
        """Handle long press action."""
        element = action.get("element")
        if not element:
            return ActionResult(False, False, "No element coordinates")

        x, y = self._convert_relative_to_absolute(element, display_width, display_height)
        device_factory = get_device_factory()
        device_factory.long_press(x, y, device_id=self.device_id)
        return ActionResult(True, False)

    def _handle_wait(self, action: dict, width: int, height: int, display_width: int, display_height: int) -> ActionResult:
        """Handle wait action."""
        duration_str = action.get("duration", "1 seconds")
        try:
            duration = float(duration_str.replace("seconds", "").strip())
        except ValueError:
            duration = 1.0

        time.sleep(duration)
        return ActionResult(True, False)

    def _handle_takeover(self, action: dict, width: int, height: int, display_width: int, display_height: int) -> ActionResult:
        """Handle takeover request (login, captcha, etc.)."""
        message = action.get("message", "User intervention required")
        self.takeover_callback(message)
        return ActionResult(True, False)

    def _handle_note(self, action: dict, width: int, height: int, display_width: int, display_height: int) -> ActionResult:
        """Handle note action (placeholder for content recording)."""
        # This action is typically used for recording page content
        # Implementation depends on specific requirements
        return ActionResult(True, False)

    def _handle_call_api(self, action: dict, width: int, height: int, display_width: int, display_height: int) -> ActionResult:
        """Handle API call action (placeholder for summarization)."""
        # This action is typically used for content summarization
        # Implementation depends on specific requirements
        return ActionResult(True, False)

    def _handle_interact(self, action: dict, width: int, height: int, display_width: int, display_height: int) -> ActionResult:
        """Handle interaction request (user choice needed)."""
        # This action signals that user input is needed
        return ActionResult(True, False, message="User interaction required")

    def _send_keyevent(self, keycode: str) -> None:
        """Send a keyevent to the device."""
        from phone_agent.hdc.connection import _run_hdc_command
        hdc_prefix = ["hdc", "-t", self.device_id] if self.device_id else ["hdc"]

        # Map common keycodes to HarmonyOS equivalents
        if keycode in ("KEYCODE_ENTER", "66", "ENTER"):
            mapped = "2054"  # HarmonyOS Enter
        else:
            mapped = str(keycode)

        try:
            _run_hdc_command(
                hdc_prefix + ["shell", "uitest", "uiInput", "keyEvent", mapped],
                capture_output=True,
                text=True,
            )
        except Exception:
            # Fallback to input keyevent for any edge cases
            subprocess.run(
                hdc_prefix + ["shell", "input", "keyevent", mapped],
                capture_output=True,
                text=True,
            )

    @staticmethod
    def _default_confirmation(message: str) -> bool:
        """Default confirmation callback using console input."""
        response = input(f"Sensitive operation: {message}\nConfirm? (Y/N): ")
        return response.upper() == "Y"

    @staticmethod
    def _default_takeover(message: str) -> None:
        """Default takeover callback using console input."""
        input(f"{message}\nPress Enter after completing manual operation...")


def parse_action(response: str) -> dict[str, Any]:
    """
    Parse action from model response.

    Args:
        response: Raw response string from the model.

    Returns:
        Parsed action dictionary.

    Raises:
        ValueError: If the response cannot be parsed.
    """
    print(f"Parsing action: {response}")
    raw = response or ""
    try:
        content = raw.strip()
        if not content:
            raise ValueError("empty action text")

        def _extract_call(src: str, start: int) -> str:
            i = start
            in_str = False
            quote = ""
            escape = False
            depth = 0
            while i < len(src) and src[i] != "(":
                i += 1
            if i >= len(src) or src[i] != "(":
                return src[start:].strip()
            depth = 1
            i += 1
            while i < len(src):
                ch = src[i]
                if escape:
                    escape = False
                elif in_str:
                    if ch == "\\":
                        escape = True
                    elif ch == quote:
                        in_str = False
                        quote = ""
                else:
                    if ch in ("'", '"'):
                        in_str = True
                        quote = ch
                    elif ch == "(":
                        depth += 1
                    elif ch == ")":
                        depth -= 1
                        if depth == 0:
                            return src[start : i + 1].strip()
                i += 1
            return src[start:].strip()

        # Extract the last <answer>...</answer> block if present
        if "<answer>" in content and "</answer>" in content:
            matches = re.findall(r"<answer>(.*?)</answer>", content, flags=re.S)
            if matches:
                content = matches[-1].strip()

        # Trim to the last actionable marker to drop <think>/其他文本
        marker_idx = max(content.rfind("finish("), content.rfind("do("))
        if marker_idx != -1:
            content = _extract_call(content, marker_idx)

        # Strip repeated leading/trailing XML tags (e.g., <answer><answer> ... </answer></answer>)
        content = re.sub(r"^(<answer>\s*)+", "", content, flags=re.I)
        content = re.sub(r"(</[a-zA-Z]+>\s*)+$", "", content)
        content = content.strip()

        if content.startswith('do(action="Type"') or content.startswith(
            'do(action="Type_Name"'
        ):
            text = content.split("text=", 1)[1][1:-2]
            return {"_metadata": "do", "action": "Type", "text": text}

        if content.startswith("do"):
            # Use AST parsing instead of eval for safety
            try:
                sanitized = (
                    content.replace("\n", "\\n")
                    .replace("\r", "\\r")
                    .replace("\t", "\\t")
                )
                tree = ast.parse(sanitized, mode="eval")
                if not isinstance(tree.body, ast.Call):
                    raise ValueError("expected a function call")

                call = tree.body
                action = {"_metadata": "do"}
                for keyword in call.keywords:
                    key = keyword.arg
                    value = ast.literal_eval(keyword.value)
                    action[key] = value
                return action
            except (SyntaxError, ValueError) as e:
                raise ValueError(f"Failed to parse do() action: {e}")

        if content.startswith("finish"):
            try:
                sanitized = (
                    content.replace("\n", "\\n")
                    .replace("\r", "\\r")
                    .replace("\t", "\\t")
                )
                tree = ast.parse(sanitized, mode="eval")
                if not isinstance(tree.body, ast.Call):
                    raise ValueError("expected a function call")

                call = tree.body
                action = {"_metadata": "finish"}
                for keyword in call.keywords:
                    key = keyword.arg
                    value = ast.literal_eval(keyword.value)
                    action[key] = value
                if "message" not in action:
                    action["message"] = ""
                return action
            except (SyntaxError, ValueError) as e:
                raise ValueError(f"Failed to parse finish() action: {e}")

        raise ValueError(f"unrecognized action text: {content[:160]}")
    except Exception as e:
        raise ValueError(f"Failed to parse action: {e} (raw={raw!r})")


def do(**kwargs) -> dict[str, Any]:
    """Helper function for creating 'do' actions."""
    kwargs["_metadata"] = "do"
    return kwargs


def finish(**kwargs) -> dict[str, Any]:
    """Helper function for creating 'finish' actions."""
    kwargs["_metadata"] = "finish"
    return kwargs
