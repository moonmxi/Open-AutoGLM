from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from phone_agent.config.apps_harmonyos import get_app_name
from .case_types import DevEcoAction, LeakCase, ScreenshotRef

_TS_IMAGE_RE = re.compile(r"^(?P<ts>\d{10,})\.(?P<ext>jpe?g|png)$", re.IGNORECASE)
_TS_JSON_RE = re.compile(r"^(?P<ts>\d{10,})\.json$", re.IGNORECASE)


@dataclass(frozen=True)
class BuildOptions:
    pre_window_s: int = 30
    post_window_s: int = 10
    max_actions: int = 12
    max_screenshots: int = 4
    suspect_k: int = 3


def _iter_event_lines(event_sequence_path: Path) -> Iterable[dict[str, Any]]:
    with event_sequence_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSONL at {event_sequence_path}:{line_no}: {e}"
                ) from e


def _normalize_action_type(action_type: str) -> str:
    action_type = (action_type or "").upper()
    mapping = {
        "CLICK": "CLICK",
        "TAP": "CLICK",
        "SWIPE": "SWIPE",
        "EDIT": "EDIT",
        "PRESS_BACK": "PRESS_BACK",
        "PRESS_ENTER": "PRESS_ENTER",
        "LONG_CLICK": "LONG_CLICK",
        "LONG_PRESS": "LONG_CLICK",
    }
    return mapping.get(action_type, action_type or "UNKNOWN")


def _flatten_actions(event: dict[str, Any], event_index: int) -> list[DevEcoAction]:
    ts_ms = int(event.get("timeStamp") or 0)
    uri = str(event.get("uri") or "")
    exact_scene_id = str(event.get("exactSceneId") or "")
    action_list = event.get("actionList") or []

    flattened: list[DevEcoAction] = []
    if not isinstance(action_list, list):
        return flattened

    for action_index, action in enumerate(action_list):
        if not isinstance(action, dict):
            continue
        action_type = _normalize_action_type(str(action.get("type") or ""))
        xpath = str(action.get("xpath") or "")
        flattened.append(
            DevEcoAction(
                ts_ms=ts_ms,
                action_type=action_type,
                xpath=xpath,
                uri=uri,
                exact_scene_id=exact_scene_id,
                event_index=event_index,
                action_index=action_index,
            )
        )
    return flattened


def _list_timestamped_screenshots(screenshot_dir: Path) -> list[tuple[int, Path]]:
    shots: list[tuple[int, Path]] = []
    if not screenshot_dir.exists():
        return shots

    for p in screenshot_dir.iterdir():
        if not p.is_file():
            continue
        m = _TS_IMAGE_RE.match(p.name)
        if not m:
            continue
        shots.append((int(m.group("ts")), p))
    shots.sort(key=lambda x: x[0])
    return shots


def _list_timestamped_layouts(layout_dir: Path) -> list[tuple[int, Path]]:
    items: list[tuple[int, Path]] = []
    if not layout_dir.exists():
        return items
    for p in layout_dir.iterdir():
        if not p.is_file():
            continue
        m = _TS_JSON_RE.match(p.name)
        if not m:
            continue
        items.append((int(m.group("ts")), p))
    items.sort(key=lambda x: x[0])
    return items


def _nearest_timestamped_path(items: list[tuple[int, Path]], ts_ms: int) -> tuple[int, Path] | None:
    if not items:
        return None
    # binary search
    lo, hi = 0, len(items) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if items[mid][0] < ts_ms:
            lo = mid + 1
        else:
            hi = mid
    candidates = [items[lo]]
    if lo - 1 >= 0:
        candidates.append(items[lo - 1])
    candidates.sort(key=lambda x: abs(x[0] - ts_ms))
    return candidates[0]


def _parse_bounds(bounds: str) -> tuple[int, int, int, int] | None:
    m = re.match(r"^\[(\-?\d+),(\-?\d+)\]\[(\-?\d+),(\-?\d+)\]$", str(bounds).strip())
    if not m:
        return None
    x1, y1, x2, y2 = map(int, m.groups())
    return x1, y1, x2, y2


def _xpath_segments(xpath: str) -> list[tuple[str, int | None]]:
    parts = [p for p in (xpath or "").split("/") if p]
    segs: list[tuple[str, int | None]] = []
    for p in parts:
        m = re.match(r"^(?P<name>[^\[]+)(\[(?P<idx>\d+)\])?$", p)
        if not m:
            segs.append((p, None))
            continue
        name = m.group("name")
        idx = int(m.group("idx")) if m.group("idx") is not None else None
        segs.append((name, idx))
    return segs


def _find_node_by_xpath(layout: dict[str, Any], xpath: str) -> dict[str, Any] | None:
    """
    Best-effort match DevEcoTesting xpath against layout tree using (type, index) path.
    """
    segs = _xpath_segments(xpath)
    if not segs:
        return None

    def node_type(node: Any) -> str | None:
        if not isinstance(node, dict):
            return None
        attrs = node.get("attributes")
        if isinstance(attrs, dict):
            return attrs.get("type") or None
        return None

    def node_id(node: Any) -> str | None:
        if not isinstance(node, dict):
            return None
        attrs = node.get("attributes")
        if isinstance(attrs, dict):
            return attrs.get("id") or None
        return None

    def node_key(node: Any) -> str | None:
        if not isinstance(node, dict):
            return None
        attrs = node.get("attributes")
        if isinstance(attrs, dict):
            return attrs.get("key") or None
        return None

    def children(node: Any) -> list[Any]:
        if not isinstance(node, dict):
            return []
        ch = node.get("children")
        return ch if isinstance(ch, list) else []

    def collect_descendant_matches(node: Any, name: str, max_depth: int = 3) -> list[Any]:
        if max_depth <= 0:
            return []
        out: list[Any] = []
        frontier = [(node, 0)]
        while frontier:
            cur, depth = frontier.pop(0)
            if depth >= max_depth:
                continue
            for ch in children(cur):
                if matches_segment(ch, name):
                    out.append(ch)
                frontier.append((ch, depth + 1))
        return out

    def matches_segment(node: Any, name: str) -> bool:
        t = node_type(node)
        if t == name:
            return True
        if node_id(node) == name or node_key(node) == name:
            return True
        return False

    # Candidate set to allow multiple index conventions.
    candidates: list[Any] = [layout]
    for name, idx in segs:
        next_candidates: list[Any] = []

        for cur in candidates:
            pool = children(cur)
            # If current itself matches, allow staying on it for this segment.
            if matches_segment(cur, name):
                pool = [cur]

            # "__Common__" is a placeholder in DevEcoTesting xpath; treat as wildcard child selector.
            if name == "__Common__":
                if not pool:
                    continue
                if idx is None:
                    next_candidates.append(pool[0])
                else:
                    for j in (idx, idx - 1):
                        if 0 <= j < len(pool):
                            next_candidates.append(pool[j])
                continue

            matched = [n for n in pool if matches_segment(n, name)]
            if not matched:
                # Some xpaths include intermediate wrappers not present in the layout tree;
                # fall back to a limited descendant search.
                matched = collect_descendant_matches(cur, name, max_depth=6)
            if not matched:
                continue

            if idx is None:
                next_candidates.append(matched[0])
                continue

            # Try multiple index conventions:
            # 1) idx among matched (0-based)
            # 2) idx-1 among matched (1-based)
            # 3) idx among all children (0-based) but must match
            # 4) idx-1 among all children (1-based) but must match
            for j in (idx, idx - 1):
                if 0 <= j < len(matched):
                    next_candidates.append(matched[j])
            for j in (idx, idx - 1):
                if 0 <= j < len(pool) and matches_segment(pool[j], name):
                    next_candidates.append(pool[j])

        # De-dup and cap to avoid explosion
        uniq: list[Any] = []
        seen_ids: set[int] = set()
        for n in next_candidates:
            nid = id(n)
            if nid in seen_ids:
                continue
            seen_ids.add(nid)
            uniq.append(n)
            if len(uniq) >= 8:
                break

        candidates = uniq
        if not candidates:
            return None

    return candidates[0] if isinstance(candidates[0], dict) else None


def _suggest_element_from_xpath(layout_path: Path, xpath: str) -> tuple[list[int], tuple[int, int, int, int]] | None:
    try:
        layout = json.loads(layout_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    target = _find_node_by_xpath(layout, xpath)
    if not target:
        # Fuzzy fallback: try to find a node whose type-path best matches the xpath suffix.
        xpath_parts = [(name, idx) for name, idx in _xpath_segments(xpath) if name and name != "__Common__"]
        if not xpath_parts:
            return None

        desired = [p[0] for p in xpath_parts]
        desired_last_idx = xpath_parts[-1][1] if xpath_parts else None

        best: dict[str, Any] | None = None
        best_score = -1
        best_area = -1
        best_parent: dict[str, Any] | None = None

        root_bounds = _parse_bounds(str((layout.get("attributes") or {}).get("bounds") or ""))
        root_area = None
        if root_bounds:
            rx1, ry1, rx2, ry2 = root_bounds
            root_area = max(1, (rx2 - rx1)) * max(1, (ry2 - ry1))

        def node_type(node: Any) -> str | None:
            if not isinstance(node, dict):
                return None
            attrs = node.get("attributes")
            if isinstance(attrs, dict):
                return attrs.get("type") or None
            return None

        def children(node: Any) -> list[Any]:
            if not isinstance(node, dict):
                return []
            ch = node.get("children")
            return ch if isinstance(ch, list) else []

        def walk(node: Any, path: list[str], parent: Any = None) -> None:
            nonlocal best, best_score, best_area, best_parent
            t = node_type(node)
            if t:
                path = path + [t]
                if t == desired[-1]:
                    # suffix match score
                    score = 0
                    for a, b in zip(reversed(desired), reversed(path)):
                        if a != b:
                            break
                        score += 1
                    attrs = node.get("attributes") if isinstance(node, dict) else None
                    if isinstance(attrs, dict):
                        b = _parse_bounds(str(attrs.get("bounds") or ""))
                        if b:
                            x1, y1, x2, y2 = b
                            area = max(0, x2 - x1) * max(0, y2 - y1)
                            # Avoid selecting huge containers as "targets"
                            if root_area is None or area <= int(root_area * 0.4):
                                if score > best_score or (
                                    score == best_score and area > best_area
                                ):
                                    best = node
                                    best_score = score
                                    best_area = area
                                    best_parent = parent
            for ch in children(node):
                walk(ch, path, node)

        walk(layout, [])
        
        # If we have a parent and an index requirement, try to pick the correct sibling
        if best and best_parent and desired_last_idx is not None:
            siblings = children(best_parent)
            best_t = node_type(best)
            same_type_siblings = [s for s in siblings if node_type(s) == best_t]
            if 0 <= desired_last_idx < len(same_type_siblings):
                best = same_type_siblings[desired_last_idx]

        # Lower threshold to 2 to allow "TabBar/Column" matches even if parent chain (Stack/Tabs) is missing
        if best is None or best_score < 2:
            return None
        target = best

    attrs = target.get("attributes") if isinstance(target, dict) else None
    if not isinstance(attrs, dict):
        return None

    def node_type(node: Any) -> str | None:
        if not isinstance(node, dict):
            return None
        a = node.get("attributes")
        if isinstance(a, dict):
            return a.get("type") or None
        return None

    def children(node: Any) -> list[Any]:
        if not isinstance(node, dict):
            return []
        ch = node.get("children")
        return ch if isinstance(ch, list) else []

    override_center: tuple[int, int] | None = None
    b = _parse_bounds(str(attrs.get("bounds") or ""))
    segs = _xpath_segments(xpath)
    col_idx = None
    for name, idx in reversed(segs):
        if name == "Column" and idx is not None:
            col_idx = idx
            break
    if col_idx is not None:
        tabbar_pos = None
        for i, (name, _) in enumerate(segs):
            if name == "TabBar":
                tabbar_pos = i
        if tabbar_pos is not None:
            prefix = "/".join(
                [
                    f"{name}[{idx}]" if idx is not None else name
                    for name, idx in segs[: tabbar_pos + 1]
                ]
            )
            tabbar = _find_node_by_xpath(layout, prefix)
            tabbar_bounds = None
            if tabbar:
                tabbar_attrs = tabbar.get("attributes") if isinstance(tabbar, dict) else None
                if isinstance(tabbar_attrs, dict):
                    tabbar_bounds = _parse_bounds(str(tabbar_attrs.get("bounds") or ""))
            if tabbar:
                columns = [c for c in children(tabbar) if node_type(c) == "Column"]
                if columns and 0 <= col_idx < len(columns):
                    col_attrs = columns[col_idx].get("attributes") if isinstance(columns[col_idx], dict) else None
                    if isinstance(col_attrs, dict):
                        col_bounds = _parse_bounds(str(col_attrs.get("bounds") or ""))
                        if col_bounds:
                            b = col_bounds
                elif tabbar_bounds:
                    x1, y1, x2, y2 = tabbar_bounds
                    count = len(columns) if columns else max(col_idx + 1, 5)
                    if x2 > x1 and count > 0:
                        cx = int(x1 + (col_idx + 0.5) * (x2 - x1) / count)
                        cy = int((y1 + y2) / 2)
                        override_center = (cx, cy)
                        b = tabbar_bounds

    if not b:
        return None
    x1, y1, x2, y2 = b
    if override_center:
        cx, cy = override_center
    else:
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

    # get root bounds for normalization
    root_bounds = _parse_bounds(str((layout.get("attributes") or {}).get("bounds") or ""))
    if not root_bounds:
        # fallback to max extents from target bounds
        rw, rh = max(x2, 1), max(y2, 1)
        rx1, ry1 = 0, 0
    else:
        rx1, ry1, rx2, ry2 = root_bounds
        rw, rh = max(rx2 - rx1, 1), max(ry2 - ry1, 1)

    rel_x = int(round(((cx - rx1) / rw) * 999))
    rel_y = int(round(((cy - ry1) / rh) * 999))
    rel_x = max(0, min(999, rel_x))
    rel_y = max(0, min(999, rel_y))

    return [rel_x, rel_y], b


def _mk_ref(ts: int, path: Path, label: str, layout_dir: Path) -> ScreenshotRef:
    layout_path = layout_dir / f"{ts}.json"
    
    return ScreenshotRef(
        ts_ms=ts, 
        path=path, 
        label=label, 
        layout_path=layout_path if layout_path.exists() else None,
        page_label=None
    )


def _pick_screenshots_around_ts(
    shots: list[tuple[int, Path]],
    leak_ts_ms: int,
    max_screenshots: int,
    layout_dir: Path,
) -> tuple[list[ScreenshotRef], dict[str, ScreenshotRef]]:
    if not shots:
        return [], {}

    # Closest before/after around the leak_ts sampling point
    pre_idx = None
    for i, (ts, _) in enumerate(shots):
        if ts <= leak_ts_ms:
            pre_idx = i
        else:
            break
    post_idx = pre_idx + 1 if pre_idx is not None else 0

    # Guardrails
    if pre_idx is None:
        pre_idx = 0
    if post_idx >= len(shots):
        post_idx = len(shots) - 1

    pre_ts, pre_path = shots[pre_idx]
    post_ts, post_path = shots[post_idx]
    key = {
        "timeline_before": _mk_ref(pre_ts, pre_path, "timeline_before", layout_dir),
        "timeline_after": _mk_ref(post_ts, post_path, "timeline_after", layout_dir),
    }

    # Extra context around pre/post
    picked_indices: list[int] = []
    picked_indices.append(pre_idx)
    picked_indices.append(post_idx)

    # Prefer 1 extra before and 1 extra after if space allows
    if len(picked_indices) < max_screenshots and pre_idx - 1 >= 0:
        picked_indices.append(pre_idx - 1)
    if len(picked_indices) < max_screenshots and post_idx + 1 < len(shots):
        picked_indices.append(post_idx + 1)

    # Fill remaining by proximity
    if len(picked_indices) < max_screenshots:
        remaining = [
            (abs(ts - leak_ts_ms), idx)
            for idx, (ts, _) in enumerate(shots)
            if idx not in set(picked_indices)
        ]
        remaining.sort(key=lambda x: x[0])
        for _, idx in remaining:
            if len(picked_indices) >= max_screenshots:
                break
            picked_indices.append(idx)

    picked_indices = sorted(set(picked_indices), key=lambda i: shots[i][0])
    refs: list[ScreenshotRef] = []
    for idx in picked_indices:
        ts, path = shots[idx]
        refs.append(_mk_ref(ts, path, label=f"context_{ts}", layout_dir=layout_dir))

    return refs, key


def _format_case_id(leak_ts_ms: int) -> str:
    dt = datetime.fromtimestamp(leak_ts_ms / 1000.0, tz=timezone.utc)
    return dt.strftime("leak_%Y%m%dT%H%M%S.%fZ")


def _extract_app_info_from_layout(layout_path: Path) -> dict[str, str | None]:
    """
    Best-effort extract app identity fields from DevEcoTesting layout JSON.

    Expected fields (when available):
      - bundleName
      - abilityName
      - pagePath
    """
    try:
        data = json.loads(layout_path.read_text(encoding="utf-8"))
    except Exception:
        return {"bundleName": None, "abilityName": None, "pagePath": None}

    def walk(node: Any) -> tuple[str | None, str | None, str | None]:
        if isinstance(node, dict):
            attrs = node.get("attributes")
            if isinstance(attrs, dict):
                bundle = attrs.get("bundleName") or None
                ability = attrs.get("abilityName") or None
                page = attrs.get("pagePath") or None
                if bundle or ability or page:
                    return str(bundle) if bundle else None, str(ability) if ability else None, str(page) if page else None
            children = node.get("children")
            if isinstance(children, list):
                for child in children:
                    b, a, p = walk(child)
                    if b or a or p:
                        return b, a, p
        elif isinstance(node, list):
            for child in node:
                b, a, p = walk(child)
                if b or a or p:
                    return b, a, p
        return None, None, None

    bundle, ability, page = walk(data)
    return {"bundleName": bundle, "abilityName": ability, "pagePath": page}


def _nearest_before_after_shots(
    shots: list[tuple[int, Path]], ts_ms: int
) -> tuple[tuple[int, Path] | None, tuple[int, Path] | None]:
    """
    Return the closest screenshot at or before ts_ms, and the closest at or after ts_ms.
    """
    if not shots:
        return None, None

    before = None
    after = None
    for ts, path in shots:
        if ts <= ts_ms:
            before = (ts, path)
        if ts >= ts_ms and after is None:
            after = (ts, path)
            break
    if before is None:
        before = shots[0]
    if after is None:
        after = shots[-1]
    return before, after


def choose_sample_leak_timestamp_ms(devecotesting_root: str | Path) -> int:
    """
    Choose a reasonable sample timestamp (ms) from the DevEcoTesting screenshot folder.

    This is meant for local simulation when you don't yet have a real memory leak timestamp.
    """
    root = Path(devecotesting_root)
    shots = _list_timestamped_screenshots(root / "screenshot")
    if not shots:
        raise FileNotFoundError(f"No timestamped screenshots found under {root / 'screenshot'}")
    return shots[len(shots) // 2][0]


def build_leak_case_from_devecotesting(
    leak_ts_ms: int,
    devecotesting_root: str | Path = "devecotesting",
    *,
    options: BuildOptions | None = None,
) -> LeakCase:
    """
    Build a LeakCase bundle by aligning DevEcoTesting artifacts around a leak timestamp.

    Intended as a stable interface for an external workflow controller script:
    - Input: leak timestamp from an external memory leak monitor tool
    - Output: a LeakCase including action window + key screenshots around that timestamp
    """
    options = options or BuildOptions()
    root = Path(devecotesting_root)
    event_sequence_path = root / "eventSequence.json"
    screenshot_dir = root / "screenshot"
    layout_dir = root / "layout"

    if not event_sequence_path.exists():
        raise FileNotFoundError(f"Missing DevEcoTesting artifact: {event_sequence_path}")

    window_start_ms = int(leak_ts_ms - options.pre_window_s * 1000)
    window_end_ms = int(leak_ts_ms + options.post_window_s * 1000)

    flattened: list[DevEcoAction] = []
    global_idx = 0
    for event_index, event in enumerate(_iter_event_lines(event_sequence_path), start=0):
        for action in _flatten_actions(event, event_index=event_index):
            if action.ts_ms < window_start_ms or action.ts_ms > window_end_ms:
                continue
            flattened.append(
                DevEcoAction(
                    **{
                        **action.to_dict(),
                        "global_index": global_idx,
                    }
                )
            )
            global_idx += 1

    # Anchor and suspects (in window indices)
    anchor_idx: int | None = None
    for i, a in enumerate(flattened):
        if a.ts_ms <= leak_ts_ms:
            anchor_idx = i
        else:
            break

    suspects: list[int] = []
    if anchor_idx is not None:
        start = max(0, anchor_idx - options.suspect_k + 1)
        suspects = list(range(start, anchor_idx + 1))

    # Trim action window for token budget
    if len(flattened) > options.max_actions:
        if anchor_idx is None:
            start_idx = 0
        else:
            start_idx = max(0, anchor_idx - (options.max_actions // 2))
        end_idx = min(len(flattened), start_idx + options.max_actions)
        start_idx = max(0, end_idx - options.max_actions)

        old_actions = flattened
        flattened = old_actions[start_idx:end_idx]

        if anchor_idx is not None:
            anchor_idx = anchor_idx - start_idx

        suspects = [i - start_idx for i in suspects if start_idx <= i < end_idx]

    shots = _list_timestamped_screenshots(screenshot_dir)
    screenshot_refs, key = _pick_screenshots_around_ts(
        shots, leak_ts_ms, options.max_screenshots, layout_dir
    )

    layouts = _list_timestamped_layouts(layout_dir)
    # Fill per-action suggested coordinates from nearest layout snapshot
    enriched_actions: list[DevEcoAction] = []
    for a in flattened:
        suggested_element = None
        suggested_bounds = None
        suggested_layout_ts_ms = None
        if a.xpath and a.action_type in ("CLICK", "EDIT", "LONG_CLICK"):
            nearest = _nearest_timestamped_path(layouts, a.ts_ms)
            if nearest is not None:
                ts, lp = nearest
                suggested = _suggest_element_from_xpath(lp, a.xpath)
                if suggested is not None:
                    suggested_element, suggested_bounds = suggested
                    suggested_layout_ts_ms = ts

        before_ref = None
        after_ref = None
        before, after = _nearest_before_after_shots(shots, a.ts_ms)
        if before is not None:
            bts, bpath = before
            before_ref = _mk_ref(bts, bpath, label=f"action_{a.global_index}_before", layout_dir=layout_dir)
        if after is not None:
            ats, apath = after
            after_ref = _mk_ref(ats, apath, label=f"action_{a.global_index}_after", layout_dir=layout_dir)

        enriched_actions.append(
            DevEcoAction(
                ts_ms=a.ts_ms,
                action_type=a.action_type,
                xpath=a.xpath,
                uri=a.uri,
                exact_scene_id=a.exact_scene_id,
                global_index=a.global_index,
                event_index=a.event_index,
                action_index=a.action_index,
                suggested_element=suggested_element,
                suggested_bounds=suggested_bounds,
                suggested_layout_ts_ms=suggested_layout_ts_ms,
                before_shot=before_ref,
                after_shot=after_ref,
            )
        )
    flattened = enriched_actions

    target_bundle_name: str | None = None
    target_ability_name: str | None = None
    target_page_path: str | None = None
    target_app_name: str | None = None
    timeline_text_hints: list[dict[str, Any]] = []

    # Strategy: 
    # Prefer extracting target info from the *start* of the action sequence (the micro-flow start),
    # rather than the leak timestamp (which might be after the user has already left the leaking page).
    # If no actions, fallback to leak_ts snapshot.
    
    layout_source_path: Path | None = None
    
    # 1. Try first action's suggested layout (closest layout to start)
    if flattened:
        first_action = flattened[0]
        if first_action.suggested_layout_ts_ms:
             # Find the path for this ts
             found = _nearest_timestamped_path(layouts, first_action.suggested_layout_ts_ms)
             if found:
                 layout_source_path = found[1]

    # 2. Fallback to timeline_before (leak_ts)
    if not layout_source_path:
        timeline_before_ref = key.get("timeline_before")
        if timeline_before_ref and timeline_before_ref.layout_path and timeline_before_ref.layout_path.exists():
            layout_source_path = timeline_before_ref.layout_path

    # Extract info if we have a source
    if layout_source_path and layout_source_path.exists():
        info = _extract_app_info_from_layout(layout_source_path)
        target_bundle_name = info.get("bundleName")
        target_ability_name = info.get("abilityName")
        target_page_path = info.get("pagePath")
        if target_bundle_name:
            target_app_name = get_app_name(target_bundle_name)

    case = LeakCase(
        case_id=_format_case_id(leak_ts_ms),
        leak_ts_ms=leak_ts_ms,
        window_start_ms=window_start_ms,
        window_end_ms=window_end_ms,
        pre_window_s=options.pre_window_s,
        post_window_s=options.post_window_s,
        max_actions=options.max_actions,
        max_screenshots=options.max_screenshots,
        suspect_k=options.suspect_k,
        source_root=root,
        event_sequence_path=event_sequence_path,
        screenshot_dir=screenshot_dir,
        layout_dir=layout_dir,
        target_bundle_name=target_bundle_name,
        target_app_name=target_app_name,
        target_ability_name=target_ability_name,
        target_page_path=target_page_path,
        actions=flattened,
        anchor_action_window_index=anchor_idx,
        suspect_action_window_indices=suspects,
        screenshots=screenshot_refs,
        key_screenshots=key,
    )
    return case



