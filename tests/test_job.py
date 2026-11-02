import pytest
from pydantic import ValidationError

from mapcreator.job import Job, WidthOption

WP = [{"lon": 147.33, "lat": -42.81}, {"lon": 147.42, "lat": -42.74}]


def test_minimal_job_defaults():
    job = Job.model_validate({"track_id": "grasstree_hill", "waypoints": WP})
    assert job.width is WidthOption.AS_MAPPED
    assert job.layouts.value == "both"


@pytest.mark.parametrize("track_id", ["Grasstree", "has space", "x" * 33, "../etc", "", "a-b"])
def test_track_id_is_a_safe_folder_name(track_id):
    with pytest.raises(ValidationError):
        Job.model_validate({"track_id": track_id, "waypoints": WP})


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        Job.model_validate({"track_id": "t", "waypoints": WP, "dem_url": "http://example.org/x.tif"})


def test_waypoint_bounds_and_counts():
    with pytest.raises(ValidationError):
        Job.model_validate({"track_id": "t", "waypoints": WP[:1]})
    with pytest.raises(ValidationError):
        Job.model_validate({"track_id": "t", "waypoints": [{"lon": 200, "lat": 0}, WP[1]]})
    with pytest.raises(ValidationError):
        Job.model_validate({"track_id": "t", "waypoints": [WP[0], WP[0]]})


def test_width_factors():
    assert WidthOption("plus_15").factor == pytest.approx(1.15)
    assert WidthOption("plus_30").factor == pytest.approx(1.30)
