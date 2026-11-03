import pytest
from pydantic import ValidationError

from mapcreator.job import Job, WidthOption, straight_line_m

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


def test_points_too_far_apart_for_the_length_limit_are_refused_before_any_build():
    perth_brisbane = [{"lon": 115.86, "lat": -31.95}, {"lon": 153.03, "lat": -27.47}]
    with pytest.raises(ValidationError, match="too far apart"):
        Job.model_validate({"track_id": "far", "waypoints": perth_brisbane})
    # 62 km in a straight line fits the 64 km limit; two points get 0.3 km of snapping slack on top.
    near = [{"lon": 147.0, "lat": -42.0}, {"lon": 147.0, "lat": -42.558}]
    assert 61_900 < straight_line_m(Job.model_validate({"track_id": "near", "waypoints": near}).waypoints) < 62_100
    far = [{"lon": 147.0, "lat": -42.0}, {"lon": 147.0, "lat": -42.6}]  # 66.7 km
    with pytest.raises(ValidationError, match="too far apart"):
        Job.model_validate({"track_id": "x", "waypoints": far})
    assert Job.model_validate({"track_id": "x", "max_length_km": 75, "waypoints": far})


def test_the_limit_is_a_fifth_more_than_targas_longest_stage():
    assert Job.model_fields["max_length_km"].default == pytest.approx(53 * 1.2, abs=0.5)
