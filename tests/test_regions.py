"""Which region, and which MGA zone, a route is built in."""

import pytest

from mapcreator.job import Waypoint
from mapcreator.regions import TASMANIA, OutOfArea, australia, job_region, mga_zone, region_for, same_region
from mapcreator.service.routing import check_region

HOBART = (147.33, -42.88)
TORQUAY = (144.33, -38.33)  # Great Ocean Road, MGA zone 55
LORNE = (143.98, -38.54)
APOLLO_BAY = (143.67, -38.76)  # zone 54
PERTH = (115.86, -31.95)


def wps(*points):
    return [Waypoint(lon=lon, lat=lat) for lon, lat in points]


@pytest.mark.parametrize(("lon", "zone"), [(113.0, 49), (115.86, 50), (138.6, 54), (144.0, 55), (147.3, 55),
                                           (151.2, 56), (153.6, 56)])
def test_mga_zone(lon, zone):
    assert mga_zone(lon) == zone


def test_tasmania_has_its_own_region_and_the_mainland_falls_back_by_zone():
    assert region_for(*HOBART) is TASMANIA
    assert region_for(*PERTH) is australia(50) and australia(50).epsg == 28350
    with pytest.raises(OutOfArea):
        region_for(174.78, -41.29)  # Wellington


def test_a_route_can_cross_a_zone_boundary_and_takes_the_zone_of_its_mean_longitude():
    assert mga_zone(TORQUAY[0]) == 55 and mga_zone(APOLLO_BAY[0]) == 54
    assert same_region([TORQUAY, LORNE]) is australia(55)  # mean 144.16
    assert same_region([LORNE, APOLLO_BAY]) is australia(54)  # mean 143.83
    assert same_region([TORQUAY, LORNE, APOLLO_BAY]) is australia(54)  # mean 143.99


def test_tasmania_and_the_mainland_never_mix():
    with pytest.raises(OutOfArea, match="same region"):
        same_region([HOBART, TORQUAY])
    with pytest.raises(OutOfArea):
        same_region([])


def test_every_stage_and_the_router_agree_on_the_region():
    points = wps(TORQUAY, LORNE, APOLLO_BAY)
    assert job_region(points) is check_region(points) is australia(54)
    assert check_region(wps(HOBART, (147.42, -42.74))) is TASMANIA
    with pytest.raises(OutOfArea, match="same region"):
        check_region(wps(HOBART, TORQUAY))
    with pytest.raises(OutOfArea, match="Point 2 is outside"):
        check_region(wps(HOBART, (174.78, -41.29)))
