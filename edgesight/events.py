

from dataclasses import dataclass

from edgesight import config


@dataclass(frozen=True)
class ZoneEvent:
    track_id: int
    action: str


class ZoneMonitor:
    def __init__(self):
        self.active_ids = set()

    def bounds(self, frame_shape):
        """Convert normalized coordinates to pixels."""

        height, width = frame_shape[:2]

        if not (
            0 <= config.ZONE_LEFT < config.ZONE_RIGHT <= 1
            and 0 <= config.ZONE_TOP < config.ZONE_BOTTOM <= 1
        ):
            raise ValueError(
                "Invalid monitoring zone configuration"
            )

        return (
            int(width * config.ZONE_LEFT),
            int(height * config.ZONE_TOP),
            int(width * config.ZONE_RIGHT),
            int(height * config.ZONE_BOTTOM),
        )

    def update(self, objects, frame_shape):
        """Check tracked objects and generate zone events."""

        left, top, right, bottom = self.bounds(
            frame_shape
        )

        current_ids = set()

        for obj in objects:
            if obj.track_id is None:
                continue

            x1, y1, x2, y2 = obj.bbox

            # Object's bottom-center position.
            center_x = (x1 + x2) / 2
            bottom_y = y2

            # Check if position falls inside zone.
            inside_zone = (
                left <= center_x <= right
                and top <= bottom_y <= bottom
            )

            if inside_zone:
                current_ids.add(obj.track_id)

        # Identify objects entering and leaving.
        entering = current_ids - self.active_ids
        exiting = self.active_ids - current_ids

        events = []

        for track_id in sorted(entering):
            events.append(
                ZoneEvent(track_id, "ENTER")
            )

        for track_id in sorted(exiting):
            events.append(
                ZoneEvent(track_id, "EXIT")
            )

        # Update tracked state for next frame.
        self.active_ids = current_ids

        status = (
            "CAUTION" if current_ids else "CLEAR"
        )

        return status, events
