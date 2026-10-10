
from edgesight.events import ZoneMonitor, ZoneEvent
from edgesight.tracker import TrackedObject


FRAME_SHAPE = (480, 640, 3)


def make_object(track_id, bbox):
    return TrackedObject(
        track_id=track_id,
        label="person",
        confidence=0.90,
        bbox=bbox,
    )


def make_inside_object(zone, track_id=7):
    """Create an object inside the current zone."""

    left, top, right, bottom = zone.bounds(
        FRAME_SHAPE
    )

    center_x = (left + right) // 2
    center_y = (top + bottom) // 2

    return make_object(
        track_id,
        (
            center_x - 20,
            center_y - 60,
            center_x + 20,
            center_y,
        ),
    )


def test_enter_event_only_once():
    zone = ZoneMonitor()

    person = make_inside_object(zone)

    status, events = zone.update(
        [person], FRAME_SHAPE
    )

    assert status == "CAUTION"
    assert events == [ZoneEvent(7, "ENTER")]

    # Remaining inside should not generate
    # another ENTER event.
    status, events = zone.update(
        [person], FRAME_SHAPE
    )

    assert status == "CAUTION"
    assert events == []


def test_exit_event():
    zone = ZoneMonitor()

    person = make_inside_object(zone)

    # First, enter the zone.
    zone.update([person], FRAME_SHAPE)

    # Then, remove the object.
    status, events = zone.update(
        [], FRAME_SHAPE
    )

    assert status == "CLEAR"
    assert events == [ZoneEvent(7, "EXIT")]
