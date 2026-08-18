from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")


class TourConfigError(ValueError):
    pass


class TourState(str, Enum):
    IDLE = "idle"
    PREPARING = "preparing"
    MOVING = "moving"
    NARRATING = "narrating"
    WAITING = "waiting_for_continue"
    PAUSING = "pausing"
    PAUSED = "paused"
    RESUMING = "resuming"
    BLOCKED = "blocked"
    STOPPING = "stopping"
    STOP_UNVERIFIED = "stop_unverified"
    SAFE_STOPPED = "safe_stopped"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class TourStop:
    stop_id: str
    waypoint_id: str
    title: str
    narration: str
    travel_announcement: str = ""


@dataclass(frozen=True, slots=True)
class TourPlan:
    schema_version: str
    tour_id: str
    title: str
    locale: str
    map_id: str
    map_version: str
    route_id: str
    stops: tuple[TourStop, ...]

    @classmethod
    def load(cls, path: str | Path) -> "TourPlan":
        resolved = Path(path)
        if resolved.stat().st_size > 1_000_000:
            raise TourConfigError("tour plan exceeds the 1 MB V0 limit")
        with resolved.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
        if not isinstance(value, dict):
            raise TourConfigError("tour plan must be a JSON object")
        return cls.from_mapping(value)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TourPlan":
        required = {
            "schema_version",
            "tour_id",
            "title",
            "locale",
            "map_id",
            "map_version",
            "route_id",
            "stops",
        }
        if set(value) != required:
            raise TourConfigError("tour plan contains missing or unexpected fields")
        if value["schema_version"] != "longship.voice-tour.v0":
            raise TourConfigError("unsupported voice-tour schema version")
        for field in required - {"stops"}:
            if not isinstance(value[field], str) or not value[field].strip():
                raise TourConfigError(f"{field} must be a non-empty string")
        for field in ("tour_id", "map_id", "map_version", "route_id"):
            if not _ID_PATTERN.fullmatch(value[field]):
                raise TourConfigError(f"{field} is not a valid public identifier")
        raw_stops = value["stops"]
        if not isinstance(raw_stops, list) or not 1 <= len(raw_stops) <= 100:
            raise TourConfigError("stops must contain between 1 and 100 entries")
        stops = tuple(_parse_stop(item) for item in raw_stops)
        if len({stop.stop_id for stop in stops}) != len(stops):
            raise TourConfigError("stop_id values must be unique")
        return cls(
            schema_version=value["schema_version"],
            tour_id=value["tour_id"],
            title=value["title"],
            locale=value["locale"],
            map_id=value["map_id"],
            map_version=value["map_version"],
            route_id=value["route_id"],
            stops=stops,
        )


def _parse_stop(value: Any) -> TourStop:
    required = {"stop_id", "waypoint_id", "title", "narration"}
    optional = {"travel_announcement"}
    if not isinstance(value, dict) or not required <= set(value):
        raise TourConfigError("every stop must contain the required fields")
    if set(value) - required - optional:
        raise TourConfigError("tour stop contains unexpected fields")
    for field in required | (optional & set(value)):
        if not isinstance(value[field], str):
            raise TourConfigError(f"stop {field} must be a string")
    if not _ID_PATTERN.fullmatch(value["stop_id"]):
        raise TourConfigError("stop_id is not a valid identifier")
    if not _ID_PATTERN.fullmatch(value["waypoint_id"]):
        raise TourConfigError("waypoint_id is not a valid identifier")
    if not value["title"].strip() or not value["narration"].strip():
        raise TourConfigError("stop title and narration must not be empty")
    if len(value["title"]) > 256 or len(value["narration"]) > 4_000:
        raise TourConfigError("stop title or narration exceeds the V0 size limit")
    if len(value.get("travel_announcement", "")) > 1_000:
        raise TourConfigError("travel announcement exceeds the V0 size limit")
    return TourStop(
        stop_id=value["stop_id"],
        waypoint_id=value["waypoint_id"],
        title=value["title"],
        narration=value["narration"],
        travel_announcement=value.get("travel_announcement", ""),
    )


@dataclass(frozen=True, slots=True)
class TourSnapshot:
    tour_id: str
    state: TourState
    revision: int
    current_stop_index: int | None
    current_stop_id: str | None
    current_waypoint_id: str | None
    total_stops: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "tour_id": self.tour_id,
            "state": self.state.value,
            "revision": self.revision,
            "current_stop_index": self.current_stop_index,
            "current_stop_id": self.current_stop_id,
            "current_waypoint_id": self.current_waypoint_id,
            "total_stops": self.total_stops,
        }
