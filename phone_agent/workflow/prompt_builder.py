from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

from .case_types import LeakCase, ScreenshotRef


# ==== Static prompt blocks (collected in one place) ====
TASK_HINT_TEMPLATE = (
    "任务简述：\n"
    "1) 场景对齐：利用提供的“动作截图”在设备上定位到起始场景。\n"
    "2) 动作复现：严格按顺序执行提供的微流程动作（3步），每一步都必须确保在对应的 Before 截图场景中执行。\n"
    "3) 自检闭环：执行完序列后，确认是否回到了起始场景（便于循环），确认无误后 finish。\n"
)

HEADER_TEMPLATE = (
    "【Leak Case Data】\n"
    "{target_lines}"
    "\n"
    "本任务提供一组导致泄漏的用户行为序列（微流程）。\n"
    "你需要根据 System Prompt 中的【核心心智模型】和【执行检查表】，在真机上复现此流程。\n"
)

# Removed redundant rule texts (SCREEN_POLICY, ACTION_PROTOCOL, ANCHOR_RULES, TIMELINE_FOCUS)
# because they are already defined in the System Prompt.

REPLAY_HINTS_HEADER = "【Replay Sequence】需复现的微流程动作序列（按时间顺序）：\n"

SCREENSHOTS_HEADER = "【Reference Screenshots】动作前后的关键参考图："
SCREENSHOTS_HINT = "提示：请利用上述截图定位元素和确认状态变化。"

ACTION_FRAME_FLOW_GUIDE = (
    "执行注意：\n"
    "- 严格按顺序复现，不要跳步。\n"
    "- 每一步操作前，必须对比当前屏幕与该步骤的 Before 截图，确认场景一致。\n"
    "- 若场景不一致，先进行导航操作（可能多步）直至对齐。\n"
    "- 每一步操作后，必须对比当前屏幕与该步骤的 After 截图，确认结果一致后再进行下一步。"
)

OUTPUT_CONTRACT_TEMPLATE = (
    "【Output Contract】\n"
    "复现成功后，使用 finish(message=...) 提交结果。\n"
    "message 必须包含：\n"
    "1) Token: <LEAK_SEQUENCE_READY>\n"
    "2) JSON 数据块：```json {{ ... }} ```，包含 case_id, leak_ts_ms, steps。\n"
    "   - steps 必须是结构化的动作列表（Tap/Swipe/Type/Back/Home/Wait），坐标为 0-999 相对坐标。\n"
    "   - 序列必须是闭环的（执行后回到起始场景）。\n"
    "JSON 示例：\n```json\n{schema_json}\n```\n"
)

def _text_part(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _image_part_from_data_url(data_url: str) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": data_url}}


def _create_user_message_from_parts(parts: list[dict[str, Any]]) -> dict[str, Any]:
    return {"role": "user", "content": parts}


def _guess_mime(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    if suffix in ("jpg", "jpeg"):
        return "image/jpeg"
    if suffix == "png":
        return "image/png"
    return "application/octet-stream"


def _image_to_data_url(
    path: Path,
    *,
    max_side: int = 512,
    jpeg_quality: int = 45,
) -> str:
    try:
        from PIL import Image

        with Image.open(path) as img:
            img = img.convert("RGB")
            w, h = img.size
            scale = max(w, h) / float(max_side) if max(w, h) > max_side else 1.0
            if scale > 1.0:
                img = img.resize((int(w / scale), int(h / scale)))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
            raw = buf.getvalue()
        mime = "image/jpeg"
    except Exception:
        raw = path.read_bytes()
        mime = _guess_mime(path)

    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _select_micro_actions(case: LeakCase, count: int = 3):
    if not case.actions:
        return []
    anchor = case.anchor_action_window_index
    if anchor is None:
        return case.actions[: max(0, min(count, len(case.actions)))]
    end = min(anchor + 1, len(case.actions))
    start = max(0, end - count)
    return case.actions[start:end]


def build_leak_case_task_hint(case: LeakCase) -> str:
    app_line = ""
    if case.target_app_name or case.target_bundle_name:
        app_line = f"目标 APP 参考：{case.target_app_name or '未知'}（{case.target_bundle_name or ''}）\n"
    return (
        app_line
        + TASK_HINT_TEMPLATE
    )


def _format_screenshots(case: LeakCase) -> str:
    lines: list[str] = []
    lines.append(SCREENSHOTS_HEADER)
    if not case.screenshots:
        lines.append("- (empty)")
        return "\n".join(lines)
    lines.append(SCREENSHOTS_HINT)
    return "\n".join(lines)


def _format_output_contract(case: LeakCase) -> str:
    schema_hint = {
        "case_id": "AUTO_FILL",
        "leak_ts_ms": 0,
        "steps": [
            {"action": "Tap", "element": ["x(0-999)", "y(0-999)"]},
            {"action": "Swipe", "start": ["x1", "y1"], "end": ["x2", "y2"]},
            {"action": "Back"},
        ],
    }
    return OUTPUT_CONTRACT_TEMPLATE.format(
        schema_json=json.dumps(schema_hint, ensure_ascii=False, indent=2)
    )


def _format_header(case: LeakCase) -> str:
    target_lines = ""
    if case.target_app_name or case.target_bundle_name:
        target_lines = (
            f"- target_app_name={case.target_app_name}\n"
            f"- target_bundle_name={case.target_bundle_name}\n"
            f"- target_ability_name={case.target_ability_name}\n"
            f"- target_page_path={case.target_page_path}\n"
        )
    return HEADER_TEMPLATE.format(
        target_lines=target_lines,
    )

def build_leak_case_extra_messages(case: LeakCase) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    
    # 1. Header
    parts.append(_text_part(_format_header(case)))
    
    # 2. Interleaved Micro-flow (Text + Images)
    actions = _select_micro_actions(case, count=3)
    if actions:
        parts.append(_text_part("【Micro-flow Actions】请严格按顺序复现以下 3 步操作：\n"))
        
        for i, act in enumerate(actions, start=1):
            # 2.1 Action Description
            target_desc = ""
            if act.xpath:
                xpath_tail = "/".join([seg for seg in act.xpath.split("/") if seg][-4:])
                if len(xpath_tail) > 80:
                    xpath_tail = "..." + xpath_tail[-80:]
                target_desc = f"xpath: {xpath_tail}"
            
            extras = []
            if act.suggested_element:
                extras.append(f"suggested_coords={act.suggested_element}")
            extra_str = f" ({', '.join(extras)})" if extras else ""

            # Prepare image info for text to reduce parts count
            ref_info_list = []
            images_to_append = []
            
            for tag, ref in (("Before", act.before_shot), ("After", act.after_shot)):
                if ref and ref.path.exists():
                    # Just use filename as requested by user
                    ref_info_list.append(f"{tag}: {ref.path.name}")
                    images_to_append.append(ref.path)
                else:
                    ref_info_list.append(f"{tag}: Missing")
            
            ref_str = ", ".join(ref_info_list)

            step_text = (
                f"Step {i}: {act.action_type}\n"
                f"  - Target: {target_desc}{extra_str}\n"
                f"  - Reference: {ref_str}"
            )
            parts.append(_text_part(step_text))

            # 2.2 Snapshots (Images only, no extra text parts)
            for img_path in images_to_append:
                parts.append(_image_part_from_data_url(
                    _image_to_data_url(img_path, max_side=360, jpeg_quality=38)
                ))
            
        parts.append(_text_part(ACTION_FRAME_FLOW_GUIDE))
    else:
        parts.append(_text_part("【Micro-flow Actions】(No actions available)"))

    # 3. Output Contract
    parts.append(_text_part(_format_output_contract(case)))

    return [_create_user_message_from_parts(parts)]
