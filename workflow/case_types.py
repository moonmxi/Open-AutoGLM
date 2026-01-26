from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DevEcoAction:
    ts_ms: int
    action_type: str
    xpath: str
    uri: str = ""
    exact_scene_id: str = ""
    global_index: int = 0
    event_index: int = 0
    action_index: int = 0
    suggested_element: list[int] | None = None  # 0-999 relative coordinates
    suggested_bounds: tuple[int, int, int, int] | None = None  # (x1,y1,x2,y2) in layout pixels
    suggested_layout_ts_ms: int | None = None
    before_shot: "ScreenshotRef | None" = None
    after_shot: "ScreenshotRef | None" = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        # Convert nested ScreenshotRef to serializable form
        for k in ("before_shot", "after_shot"):
            ref = payload.get(k)
            if isinstance(ref, ScreenshotRef):
                payload[k] = ref.to_dict()
        return payload


@dataclass(frozen=True)
class ScreenshotRef:
    ts_ms: int
    path: Path
    label: str
    layout_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["path"] = str(self.path)
        payload["layout_path"] = str(self.layout_path) if self.layout_path else None
        return payload


@dataclass
class LeakCase:
    case_id: str
    leak_ts_ms: int
    window_start_ms: int
    window_end_ms: int

    pre_window_s: int
    post_window_s: int
    max_actions: int
    max_screenshots: int
    suspect_k: int

    source_root: Path
    event_sequence_path: Path
    screenshot_dir: Path
    layout_dir: Path

    target_bundle_name: str | None = None
    target_app_name: str | None = None
    target_ability_name: str | None = None
    target_page_path: str | None = None
    pre_leak_text_hints: list[str] = field(default_factory=list)
    post_leak_text_hints: list[str] = field(default_factory=list)

    actions: list[DevEcoAction] = field(default_factory=list)
    anchor_action_window_index: int | None = None
    suspect_action_window_indices: list[int] = field(default_factory=list)

    screenshots: list[ScreenshotRef] = field(default_factory=list)
    key_screenshots: dict[str, ScreenshotRef] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "leak_ts_ms": self.leak_ts_ms,
            "window_start_ms": self.window_start_ms,
            "window_end_ms": self.window_end_ms,
            "pre_window_s": self.pre_window_s,
            "post_window_s": self.post_window_s,
            "max_actions": self.max_actions,
            "max_screenshots": self.max_screenshots,
            "suspect_k": self.suspect_k,
            "source_root": str(self.source_root),
            "event_sequence_path": str(self.event_sequence_path),
            "screenshot_dir": str(self.screenshot_dir),
            "layout_dir": str(self.layout_dir),
            "target_bundle_name": self.target_bundle_name,
            "target_app_name": self.target_app_name,
            "target_ability_name": self.target_ability_name,
            "target_page_path": self.target_page_path,
            "pre_leak_text_hints": list(self.pre_leak_text_hints),
            "post_leak_text_hints": list(self.post_leak_text_hints),
            "actions": [a.to_dict() for a in self.actions],
            "anchor_action_window_index": self.anchor_action_window_index,
            "suspect_action_window_indices": list(self.suspect_action_window_indices),
            "screenshots": [s.to_dict() for s in self.screenshots],
            "key_screenshots": {
                k: v.to_dict() for k, v in self.key_screenshots.items()
            },
        }
