from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

from .case_types import LeakCase, ScreenshotRef


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


def build_leak_case_task_hint(case: LeakCase) -> str:
    app_line = ""
    if case.target_app_name or case.target_bundle_name:
        app_line = f"目标 APP 参考：{case.target_app_name or '未知'}（{case.target_bundle_name or ''}）\n"
    return (
        app_line
        + "三阶段要求：\n"
        + "1) 用实时截图对齐到 pre_leak 目标界面（参考目标前两张截图+pre_leak，必要时 Home->Launch）。\n"
        + "2) 在目标界面内按 Actions Window / Replay Hints 复现压力动作（首轮），完成后先回到首轮开始的场景。\n"
        + "3) 自测：按自己构造的同一序列再执行一轮，确认无误后，用 <LEAK_SEQUENCE_READY> + ```json``` finish；JSON 至少含 case_id / leak_ts_ms / steps，steps 不含 Launch，执行完必须回到序列开始场景。\n"
    )


def _format_actions_window(case: LeakCase) -> str:
    lines: list[str] = []
    total = len(case.actions)
    lines.append("【Actions Window】已按时间排序；至少覆盖 MUST_REPLAY 的两步，再组合可重复动作序列")
    if not case.actions:
        lines.append("- (empty)")
        return "\n".join(lines)

    anchor = case.anchor_action_window_index
    indices: set[int] = set(case.suspect_action_window_indices)
    if anchor is not None:
        indices.update({anchor, max(0, anchor - 1), min(total - 1, anchor + 1)})
    if not indices:
        indices = set(range(min(6, total)))

    shown = sorted(indices)
    lines.append(f"- showing {len(shown)} / {total}，至少包含 MUST_REPLAY 的两步，再组合可重复动作序列")

    must_replay_indices: set[int] = set()
    pre_actions = [i for i, a in enumerate(case.actions) if a.ts_ms <= case.leak_ts_ms]
    if pre_actions:
        last_two = pre_actions[-2:] if len(pre_actions) >= 2 else pre_actions[-1:]
        must_replay_indices.update(last_two)

    for i in shown:
        a = case.actions[i]
        delta_ms = case.leak_ts_ms - a.ts_ms
        marks: list[str] = []
        if anchor == i:
            marks.append("ANCHOR")
        if i in set(case.suspect_action_window_indices):
            marks.append("SUSPECT")
        if i in must_replay_indices:
            marks.append("MUST_REPLAY")
        mark = f" [{'|'.join(marks)}]" if marks else ""

        xpath_tail = "/".join([seg for seg in a.xpath.split("/") if seg][-4:]) if a.xpath else ""
        if len(xpath_tail) > 160:
            xpath_tail = "..." + xpath_tail[-160:]
        scene = a.exact_scene_id[:8] if a.exact_scene_id else ""
        scene_part = f" scene={scene}" if scene else ""
        snap_part = ""
        if a.before_shot or a.after_shot:
            before_name = a.before_shot.path.name if a.before_shot else ""
            after_name = a.after_shot.path.name if a.after_shot else ""
            snap_part = f" shots(before={before_name or '-'}, after={after_name or '-'})"
        sugg = f" suggested_element={a.suggested_element}" if a.suggested_element else ""

        lines.append(
            f"- #{i}{mark} ts={a.ts_ms} (delta_ms={delta_ms}) type={a.action_type}{scene_part}{sugg} xpath_tail={xpath_tail}{snap_part}"
        )
    return "\n".join(lines)


def _format_replay_hints(case: LeakCase) -> str:
    if not case.actions:
        return "【Replay Hints】无可参考动作"
    before = [a for a in case.actions if a.ts_ms <= case.leak_ts_ms]
    after = [a for a in case.actions if a.ts_ms > case.leak_ts_ms]
    selected = before[-3:] + after[:2]

    def fmt_action(a: Any) -> str:
        scene = a.exact_scene_id[:8] if a.exact_scene_id else ""
        scene_part = f" scene={scene}" if scene else ""
        xpath_tail = "/".join([seg for seg in a.xpath.split("/") if seg][-4:]) if a.xpath else ""
        if len(xpath_tail) > 120:
            xpath_tail = "..." + xpath_tail[-120:]
        return f"- ts={a.ts_ms} type={a.action_type}{scene_part} xpath_tail={xpath_tail}"

    return "【Replay Hints】列出漏点前3步 + 后2步，复现时优先覆盖前三步\n" + "\n".join(fmt_action(a) for a in selected)


def _format_screenshots(case: LeakCase) -> str:
    lines: list[str] = []
    lines.append("【Screenshots Window】窗口内截图，文件名为时间戳；包含 pre/post 及少量关键参考图（控制 token）")
    if not case.screenshots:
        lines.append("- (empty)")
        return "\n".join(lines)
    for s in case.screenshots:
        delta_ms = s.ts_ms - case.leak_ts_ms
        lines.append(f"- {s.label}: ts={s.ts_ms} (delta_ms={delta_ms}) file={s.path.name}")
    return "\n".join(lines)


def _format_output_contract(case: LeakCase) -> str:
    schema_hint = {
        "case_id": case.case_id,
        "leak_ts_ms": case.leak_ts_ms,
        "steps": [
            {"action": "Tap", "element": [123, 456]},
            {"action": "Swipe", "start": [500, 800], "end": [500, 200]},
            {"action": "Back"},
        ],
    }
    return (
        "【Output Contract】\n"
        "当你认为已复现且可回放时，请使用 finish(message=...) 收口，并满足：\n"
        "1) 必须用单引号包裹 message：finish(message='...')（防止 JSON 被截断）\n"
        '2) message 必须包含 token: "<LEAK_SEQUENCE_READY>"\n'
        "3) message 必须包含一段 ```json ... ```，字段至少有 case_id / leak_ts_ms / steps\n"
        "4) steps 为结构化动作列表（不要放 do(...) 字符串），字段与 ActionHandler 一致（Tap/Swipe/Type/Back/Home/Wait 等，不要包含 Launch），坐标使用 0-999 相对坐标系；执行完 steps 后应回到执行 steps 前的场景，便于可重复回放\n"
        f"JSON 示例：\n```json\n{json.dumps(schema_hint, ensure_ascii=False, indent=2)}\n```\n"
    )


def build_leak_case_text(case: LeakCase) -> str:
    header = (
        f"【Leak Case】case_id={case.case_id}\n"
        f"- T_leak(ms)={case.leak_ts_ms}\n"
        f"- window=[{case.window_start_ms}, {case.window_end_ms}] (pre={case.pre_window_s}s, post={case.post_window_s}s)\n"
        f"- suspects_k={case.suspect_k}, max_actions={case.max_actions}, max_screenshots={case.max_screenshots}\n"
        + (
            f"- target_app_name={case.target_app_name}\n"
            f"- target_bundle_name={case.target_bundle_name}\n"
            f"- target_ability_name={case.target_ability_name}\n"
            f"- target_page_path={case.target_page_path}\n"
            if (case.target_app_name or case.target_bundle_name)
            else ""
        )
        + "\n"
        "任务：结合“动作窗口”和“关键截图”，在设备上找到对应场景并复现导致泄漏的操作路径。\n"
        "要求：\n"
        "1) 把 pre_leak 截图当作目标界面，优先导航到与之匹配的场景；\n"
        "2) 继续覆盖 SUSPECT/ANCHOR 动作，避免与泄漏无关的探索；\n"
        "3) 序列必须闭环可重复：推荐 Home->Launch 起步并进入目标场景后再复现；若中途偏航先 Back/Home 归位，不要把误点/探索动作写入 steps；steps 不含 Launch；执行完 steps 后应回到执行 steps 前的场景（不含 Launch），便于循环压力测试；\n"
        "4) 最终对齐到与 post_leak 相近的界面；\n"
        "5) 只做场景对齐与路径复现，不要尝试读取内存/接口/设置。\n"
    )

    hints: list[str] = []
    if case.pre_leak_text_hints:
        hints.append("【pre_leak UI 文本线索】" + " / ".join(case.pre_leak_text_hints[:16]))
    if case.post_leak_text_hints:
        hints.append("【post_leak UI 文本线索】" + " / ".join(case.post_leak_text_hints[:16]))

    parts = [
        header,
        "\n".join(hints).strip(),
        _format_actions_window(case),
        "",
        _format_replay_hints(case),
        "",
        _format_screenshots(case),
        "",
        _format_output_contract(case),
    ]
    return "\n".join([p for p in parts if p]).strip() + "\n"


def build_leak_case_extra_messages(case: LeakCase) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    parts.append(_text_part(build_leak_case_text(case)))

    pre = [s for s in case.screenshots if s.ts_ms <= case.leak_ts_ms]
    post = [s for s in case.screenshots if s.ts_ms >= case.leak_ts_ms]

    selected: list[ScreenshotRef] = []
    if pre:
        selected.extend(pre[-3:])
    if post:
        selected.append(post[0])

    pre_key = case.key_screenshots.get("pre_leak")
    post_key = case.key_screenshots.get("post_leak")

    seen_ts: set[int] = set()
    for ref in selected:
        if ref.ts_ms in seen_ts:
            continue
        seen_ts.add(ref.ts_ms)

        label = ref.label
        if pre_key and ref.ts_ms == pre_key.ts_ms:
            label = "pre_leak"
        if post_key and ref.ts_ms == post_key.ts_ms:
            label = "post_leak"
        parts.append(_text_part(f"【Reference screenshot】label={label} ts={ref.ts_ms} file={ref.path.name}（按时间顺序导航，优先用于阶段1 场景对齐）"))
        parts.append(_image_part_from_data_url(_image_to_data_url(ref.path, max_side=384, jpeg_quality=40)))

    # Attach action-specific before/after snapshots for anchor/suspect actions
    candidate_indices: list[int] = []
    if case.anchor_action_window_index is not None:
        candidate_indices.append(case.anchor_action_window_index)
    candidate_indices.extend(case.suspect_action_window_indices)
    # Always include MUST_REPLAY (the last two actions at/before leak_ts_ms) so the model
    # can see before/after snapshots of the most critical recorded steps.
    pre_actions = [i for i, a in enumerate(case.actions) if a.ts_ms <= case.leak_ts_ms]
    if pre_actions:
        candidate_indices.extend(pre_actions[-2:] if len(pre_actions) >= 2 else pre_actions[-1:])
    candidate_indices = sorted({i for i in candidate_indices if 0 <= i < len(case.actions)})[:4]

    max_images = 6
    images_used = 0
    for idx in candidate_indices:
        if images_used >= max_images:
            break
        a = case.actions[idx]
        for tag, ref in (("before", a.before_shot), ("after", a.after_shot)):
            if ref is None or not ref.path.exists():
                continue
            parts.append(_text_part(f"【Action snapshot】idx={idx} ts={a.ts_ms} tag={tag} file={ref.path.name}（帮助推断CLICK目标前后变化）"))
            parts.append(_image_part_from_data_url(_image_to_data_url(ref.path, max_side=360, jpeg_quality=38)))
            images_used += 1
            if images_used >= max_images:
                break

    return [_create_user_message_from_parts(parts)]
