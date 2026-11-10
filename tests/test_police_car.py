"""Police cars: every jurisdiction builds, in its frame, within budget, from drawn textures; placement; the picker."""

import builtins
from types import SimpleNamespace

import numpy as np
import pytest
import shapely
from PIL import Image

from mapcreator.core import kn5, police_car, textures
from mapcreator.core.police_car import CELLS, JURISDICTIONS
from mapcreator.core.scene import MAX_VERTICES

BUDGET_VERTICES = 700  # the intersections report's estimate for a car
BUDGET_TRIANGLES = 400


def all_vertices(car):
    pos = np.vstack([m.positions for m in car.meshes]).astype(float)
    nrm = np.vstack([m.normals for m in car.meshes]).astype(float)
    uv = np.vstack([m.uvs for m in car.meshes]).astype(float)
    return pos, nrm, uv


def in_cell(uv, cell):
    x0, y0, x1, y1 = (v / police_car.ATLAS for v in CELLS[cell])
    return (uv[:, 0] >= x0) & (uv[:, 0] <= x1) & (uv[:, 1] >= y0) & (uv[:, 1] <= y1)


def winding_ok(mesh) -> bool:
    p = mesh.positions.astype(float)
    t = mesh.triangles
    cross = np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]])
    return bool(((cross * mesh.normals[t].astype(float).mean(axis=1)).sum(axis=1) >= 0).all())


@pytest.mark.parametrize("code", JURISDICTIONS)
def test_every_jurisdiction_builds(code):
    car = police_car.police_car(code)
    assert car.jurisdiction == code
    for m in car.meshes + [car.collider]:
        m.validate()
    assert all(m.renderable for m in car.meshes) and not car.collider.renderable
    assert car.collider.name == police_car.COLLIDER_MESH
    assert all(m.name.startswith(f"POLICE_{code.upper()}_") and not m.name[0].isdigit() for m in car.meshes)
    names = police_car.material_names(code)
    assert {m.material for m in car.meshes} == set(car.materials) == set(names.values())
    # One texture for all three materials, and the night list names the band and the lenses.
    assert {mat.texture for mat in car.materials.values()} == set(car.textures) == {names["body"]}
    assert car.textures[names["body"]].size == (police_car.ATLAS, police_car.ATLAS)
    assert car.night_diffuse == {names["body"]: 0.35, names["retro"]: 1.3, names["lens"]: 2.0}
    assert set(car.kn5_properties) == set(car.materials)
    assert all(props["ksDiffuse"] == 0.6 for props in car.kn5_properties.values())  # the walls' value, by day


@pytest.mark.parametrize("code", JURISDICTIONS)
@pytest.mark.parametrize("body", sorted(police_car.BODIES))
def test_budget(code, body):
    car = police_car.police_car(code, body)
    assert car.vertices <= BUDGET_VERTICES
    assert car.triangles <= BUDGET_TRIANGLES


@pytest.mark.parametrize("body", sorted(police_car.BODIES))
def test_local_frame(body):
    """Origin at the car's centre on the ground, +z forward, +x right, +y up, metres."""
    car = police_car.police_car("tas", body)
    shape = car.body
    pos, nrm, uv = all_vertices(car)
    lo, hi = pos.min(axis=0), pos.max(axis=0)
    assert lo[1] == pytest.approx(0.0, abs=1e-6)  # the tyres stand on the ground
    assert hi[1] == pytest.approx(shape.height, abs=0.02)  # the light bar is the top
    assert lo[0] == pytest.approx(-hi[0], abs=1e-6) and lo[2] == pytest.approx(-hi[2], abs=1e-6)  # centred
    assert hi[2] - lo[2] == pytest.approx(shape.length, abs=1e-6)
    assert 4.6 < shape.length < 5.0 and 1.8 < shape.width < 1.95
    # The faces drawn from each view are on that side of the car: the front view's at +z, the right side's at +x.
    assert pos[in_cell(uv, "front"), 2].min() > shape.length / 2 - 0.01
    assert pos[in_cell(uv, "rear"), 2].max() < -shape.length / 2 + 0.01
    assert pos[in_cell(uv, "side_r"), 0].min() > 0.5 and pos[in_cell(uv, "side_l"), 0].max() < -0.5
    # The bar's lenses sit on the roof; the windscreen leans back from the bonnet.
    lens = next(m for m in car.meshes if m.material.endswith("_lens"))
    assert lens.positions[:, 1].min() >= shape.roof - 0.01
    glass = pos[in_cell(uv, "glass") & (nrm[:, 1] > 0.3) & (pos[:, 2] > -0.5)]
    assert np.corrcoef(glass[:, 2], glass[:, 1])[0, 1] < -0.9  # higher further back
    for m in car.meshes:
        assert winding_ok(m)


@pytest.mark.parametrize("code", JURISDICTIONS)
def test_the_band_is_its_own_reflective_strip(code):
    """Along the doors the band's faces run exactly between its heights; towards the ends it drops with the belt
    (a wedge keeps its foot), never above them."""
    car = police_car.police_car(code)
    retro = next(m for m in car.meshes if m.material.endswith("_retro"))
    b0, b1 = police_car.band_heights(car.body, police_car.LIVERIES[code])
    side = np.abs(retro.normals[:, 0]) > 0.7
    assert side.mean() > 0.6
    y, z = retro.positions[side, 1].astype(float), retro.positions[side, 2].astype(float)
    doors = np.abs(z) < 0.3  # the station at the windscreen's top, under the roof
    assert np.allclose(np.sort(np.unique(np.round(y[doors], 4))), [round(b0, 4), round(b1, 4)], atol=1e-3)
    assert y.max() <= b1 + 1e-3


COLLIDER_TOL_M = 0.045  # "within a few cm": its sides stand 1 cm in from the band, so 4 cm out from the sill's edge


def collider_groups(mesh):
    """The collider's pieces: its triangles grouped by the vertices they share."""
    parent = list(range(len(mesh.positions)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b, c in mesh.triangles:
        parent[root(b)] = root(a)
        parent[root(c)] = root(a)
    groups = {}
    for t in mesh.triangles:
        groups.setdefault(root(t[0]), []).append(t)
    return [np.array(g) for g in groups.values()]


@pytest.mark.parametrize("body", sorted(police_car.BODIES))
def test_collider_is_the_body_in_convex_pieces(body):
    """No invisible face: the collider follows the bonnet, roof line, boot and sides, in three convex pieces."""
    car = police_car.police_car("tas", body)
    shape, c = car.body, car.collider
    c.validate()
    assert c.name == police_car.COLLIDER_MESH and not c.renderable
    assert np.allclose(np.linalg.norm(c.normals, axis=1), 1, atol=1e-4)
    p = c.positions.astype(float)
    groups = collider_groups(c)
    assert len(groups) == 3  # bonnet, cabin, boot
    for tris in groups:
        verts = np.unique(tris)
        centre = p[verts].mean(axis=0)
        for a, b, d in tris:
            n = np.cross(p[b] - p[a], p[d] - p[a])
            n /= np.linalg.norm(n)
            assert n @ (p[[a, b, d]].mean(axis=0) - centre) > 0  # wound outwards
            assert ((p[verts] - p[a]) @ n <= 1e-5).all()  # convex: every corner on the inner side of every face
    # Within the body's silhouette, station by station (the collider's corners are all on stations).
    for x, y, z in p:
        st = next(s for s in shape.stations if abs(s.z - z) < 1e-5)
        section = shapely.Polygon(police_car._section(st, police_car.band_heights(shape, police_car.LIVERIES["tas"]),
                                                      False))
        assert section.buffer(COLLIDER_TOL_M).contains(shapely.Point(x, y)), (x, y, z)
    # No higher than the bonnet and boot at the nose and tail, and the roof's height over the cabin.
    nose, tail = shape.stations[0], shape.stations[-1]
    assert p[np.abs(p[:, 2] - nose.z) < 1e-5, 1].max() == pytest.approx(nose.belt)
    assert p[np.abs(p[:, 2] - tail.z) < 1e-5, 1].max() == pytest.approx(tail.belt)
    assert p[:, 1].max() == pytest.approx(shape.roof)
    # And it covers the body: every corner of the drawn body and band (not the wheels, the mirrors, or the light bar
    # 9 cm above the roof) is inside a piece.
    drawn = [m for m in car.meshes if not m.material.endswith("_lens")]
    pos = np.vstack([m.positions for m in drawn]).astype(float)
    uv = np.vstack([m.uvs for m in drawn]).astype(float)
    pos = pos[~(in_cell(uv, "wheel") | in_cell(uv, "tread"))]
    hw = np.interp(pos[:, 2], [s.z for s in shape.stations][::-1], [s.half_w for s in shape.stations][::-1])
    pos = pos[np.abs(pos[:, 0]) <= hw + 1e-6]  # the mirrors stand out beyond the body; nothing else does
    planes = []
    for tris in groups:
        n = np.cross(p[tris[:, 1]] - p[tris[:, 0]], p[tris[:, 2]] - p[tris[:, 0]])
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        planes.append((n, (n * p[tris[:, 0]]).sum(axis=1)))
    inside = np.zeros(len(pos), bool)
    for n, d in planes:
        inside |= ((pos @ n.T - d) <= COLLIDER_TOL_M).all(axis=1)
    assert inside.all(), pos[~inside][:5]


@pytest.mark.parametrize("body", sorted(police_car.BODIES))
def test_collider_budget(body):
    c = police_car.police_car("vic", body).collider
    assert len(c.positions) <= 64 and len(c.triangles) <= 120


def test_place_maps_the_car_into_ac_axes():
    car = police_car.police_car("qld")
    at = np.array([100.0, 20.0, -50.0])
    # Facing east (+x in AC): its right is south (+z in AC, since z = -north).
    placed = police_car.place(car, at, (1.0, 0.0, 0.0), index=3)
    assert [m.name for m in placed] == [m.name + "_3" for m in car.meshes] + [police_car.COLLIDER_MESH + "_3"]
    assert not placed[-1].renderable
    for local, world in zip(car.meshes + [car.collider], placed):
        lp, wp = local.positions.astype(float), world.positions.astype(float)
        assert np.allclose(wp[:, 0], at[0] + lp[:, 2], atol=1e-4)  # forward -> east
        assert np.allclose(wp[:, 2], at[2] + lp[:, 0], atol=1e-4)  # right -> south
        assert np.allclose(wp[:, 1], at[1] + lp[:, 1], atol=1e-4)
        assert winding_ok(world)
    # Tilted on a slope, still wound the KN5's way, and facing along the slope.
    up = np.array([0.0, 1.0, 0.1]) / np.linalg.norm([0.0, 1.0, 0.1])
    for m in police_car.place(car, at, (1.0, 0.0, 0.0), up=up):
        assert winding_ok(m)


def test_place_keeps_the_lettering_readable():
    """Placed in AC, a viewer on the car's right looks at the right side's view, whose front is on the right:
    the car isn't mirrored."""
    car = police_car.police_car("sa")
    (body, *_) = police_car.place(car, (0.0, 0.0, 0.0), (0.0, 0.0, -1.0))  # facing north; its right is east (+x)
    uv = body.uvs.astype(float)
    side = in_cell(uv, "side_r")
    assert body.positions[side, 0].min() > 0.5
    # In the right side's cell u grows towards the front, which is north (-z) here.
    p = body.positions[side].astype(float)
    assert np.corrcoef(uv[side, 0], -p[:, 2])[0, 1] > 0.99


def test_merge_joins_six_cars_by_material():
    car = police_car.police_car("wa")
    placed = [m for k in range(6) for m in police_car.place(car, (k * 10.0, 0.0, 0.0), (0.0, 0.0, 1.0), index=k)]
    merged = police_car.merge(placed)
    assert len(merged) == len(car.meshes) + 1
    assert len({m.name for m in merged}) == len(merged)
    assert sum(len(m.positions) for m in merged) == 6 * (car.vertices + len(car.collider.positions))
    for m in merged:
        m.validate()
        assert len(m.positions) <= MAX_VERTICES
        assert winding_ok(m)
    assert sum(not m.renderable for m in merged) == 1


def test_six_cars_cost_in_a_kn5(tmp_path):
    """Up to six cars a track (the intersections report's Q7), sharing one texture: the KN5's cost."""
    car = police_car.police_car("tas")
    placed = [m for k in range(6) for m in police_car.place(car, (k * 10.0, 0.0, 0.0), (0.0, 0.0, 1.0), index=k)]
    meshes = police_car.merge(placed)
    tex = {f"{name}.dds": textures.to_dds(im) for name, im in car.textures.items()}
    mats = [kn5.Kn5Material(name, properties=car.kn5_properties[name], textures={"txDiffuse": f"{m.texture}.dds"})
            for name, m in car.materials.items()]
    index = {m.name: i for i, m in enumerate(mats)}
    kmeshes = [kn5.Kn5Mesh(m.name, index[m.material], m.positions, m.normals, m.uvs, m.triangles,
                           renderable=m.renderable) for m in meshes]
    with_cars = tmp_path / "cars.kn5"
    kn5.write(with_cars, root="t", textures=tex, materials=mats, meshes=kmeshes, dummies=[])
    texture_bytes = sum(len(b) for b in tex.values())
    assert texture_bytes < 720_000  # one 1024 x 1024 DXT1 atlas with its mips
    assert with_cars.stat().st_size - texture_bytes < 250_000  # the geometry of six cars and their colliders
    assert kn5.read(with_cars)["nodes"]


def test_textures_are_drawn_not_loaded(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError(f"tried to open {args[:1]}")

    police_car.atlas.cache_clear()
    monkeypatch.setattr(Image, "open", refuse)
    monkeypatch.setattr(builtins, "open", refuse)
    for code in JURISDICTIONS:
        car = police_car.police_car(code)
        im = car.textures[police_car.material_names(code)["body"]]
        assert im.mode == "RGB" and len(np.unique(np.asarray(im.resize((32, 32))).reshape(-1, 3), axis=0)) > 8


def test_each_livery_looks_different():
    a = {code: np.asarray(police_car.atlas(code, police_car.LIVERIES[code].body), float) for code in JURISDICTIONS}
    for i, x in enumerate(JURISDICTIONS):
        for y in JURISDICTIONS[i + 1:]:
            if a[x].shape == a[y].shape:
                assert np.abs(a[x] - a[y]).mean() > 1.0, (x, y)


def test_a_second_body_has_its_own_names():
    default = police_car.LIVERIES["nsw"].body
    other = next(b for b in police_car.BODIES if b != default)
    car = police_car.police_car("nsw", other)
    assert set(car.materials) == set(police_car.material_names("nsw", other).values())
    assert all(other in name for name in car.materials)
    assert police_car.police_car("nsw").textures.keys().isdisjoint(car.textures)
    with pytest.raises(ValueError):
        police_car.police_car("xyz")


def test_tasmania_reads_silver_and_the_rest_white():
    """In sunlight the first silver (195) rendered as white (in game, 10 Nov 2026): it's darker now, with more sheen."""
    tas = police_car.LIVERIES["tas"]
    assert max(tas.paint) <= 175 and tas.metallic
    assert all(police_car.LIVERIES[c].paint == police_car.WHITE for c in JURISDICTIONS if c != "tas")
    assert np.mean(police_car.WHITE) - np.mean(tas.paint) >= 50
    silver = police_car.police_car("tas").kn5_properties["police_tas"]
    white = police_car.police_car("vic").kn5_properties["police_vic"]
    assert silver["ksSpecular"] > 2 * white["ksSpecular"] and silver["ksSpecularEXP"] < white["ksSpecularEXP"]


def test_night_values_keep_the_body_and_lenses_from_washing_out():
    """At 6 m in the beam the body (0.6, unadjusted) clipped to white, the band at 2.0 lost its chequer and the lenses
    at 4.0 burned white (in game, 10 Nov 2026)."""
    night = police_car.NIGHT_DIFFUSE_BY_PART
    day = police_car.PAINT_PROPS["ksDiffuse"]
    assert night["body"] < day < night["retro"] < 2.0 and day < night["lens"] <= 2.0
    # Little specular on the paint and the band: with the headlights beside the eye, a side facing them catches it.
    assert police_car.PAINT_PROPS["ksSpecular"] <= 0.15
    assert max(police_car.WHITE) <= 230


def test_the_light_bar_is_drawn_switched_off():
    def lum(c):
        return float(np.dot(c, [0.2126, 0.7152, 0.0722]))

    for colour in (police_car.RED, police_car.BLUE):
        tint = police_car.lens_tint(colour)
        assert lum(tint) < lum(colour) and max(tint) < 0.75 * max(colour)
    im = np.asarray(police_car.atlas("vic", "suv"), float)
    x0, y0, x1, y1 = CELLS["bar_front"]
    assert im[y0:y1, x0:x1].max(axis=2).max() < 200  # nothing near white, even the clear lens


@pytest.mark.parametrize("code", JURISDICTIONS)
def test_crests_are_drawn_large(code):
    livery = police_car.LIVERIES[code]
    size = livery.crest_side[2] if livery.crest_side else livery.crest_bonnet
    assert livery.crest in police_car.CRESTS and size >= 0.30


def test_lettering_and_crests_keep_their_size_in_metres():
    """The cells aren't square in metres (the sedan's side is about 405 px a metre along and 547 up), so lettering is
    sized by each axis: the same word comes out the same height and proportions in any view."""
    shape = police_car.SUV
    ratios = []
    for name in ("side_r", "front", "top"):
        view = police_car.views(shape)[name]
        x0, y0, x1, y1 = (v * police_car.SUPERSAMPLE for v in CELLS[name])
        img = Image.new("RGB", (x1 - x0, y1 - y0), (255, 255, 255))
        a, b = (view.a0 + view.a1) / 2, (view.b0 + view.b1) / 2
        rotate = 90 if name == "top" else 0
        police_car._label(img, view, "Police", a, b, 0.08, (0, 0, 0), rotate=rotate)
        ys, xs = np.nonzero(np.asarray(img.convert("L")) < 128)
        sa, sb = police_car._scale(view, 0), police_car._scale(view, 1)
        across, up = (xs.max() - xs.min()) / sa, (ys.max() - ys.min()) / sb
        cap, length = (up, across) if rotate == 0 else (across, up)
        assert cap == pytest.approx(0.08, rel=0.08)
        ratios.append(length / cap)
    assert max(ratios) / min(ratios) < 1.08
    # A crest is square in metres, so drawn into a 100 x 60 px box it's stretched by exactly that against a square.
    def aspect(w, h):
        img = Image.new("RGB", (200, 200), (255, 255, 255))
        police_car.draw_crest(img, "qld", 100, 100, w, h)
        ys, xs = np.nonzero(np.asarray(img.convert("L")) < 250)
        return (xs.max() - xs.min()) / (ys.max() - ys.min())

    assert aspect(100, 60) / aspect(100, 100) == pytest.approx(100 / 60, rel=0.06)


def test_nsw_emergency_number_is_large_enough_to_read():
    """At 0.045 m it wasn't legible at 6 m; Victoria's "131 444", at about 0.09 m, was (in game, 10 Nov 2026)."""
    (text, _, _, cap), = police_car.LIVERIES["nsw"].extra
    assert text == "000 EMERGENCY" and cap >= 0.08


@pytest.mark.parametrize("lon, lat, code", [
    (147.33, -42.88, "tas"),  # Hobart
    (144.10, -39.90, "tas"),  # King Island
    (151.21, -33.87, "nsw"),  # Sydney
    (147.37, -35.12, "nsw"),  # Wagga Wagga
    (153.565, -28.866, "nsw"),  # Ballina
    (144.96, -37.81, "vic"),  # Melbourne
    (144.28, -36.76, "vic"),  # Bendigo
    (146.31, -39.00, "vic"),  # Wilsons Promontory
    (153.03, -27.47, "qld"),  # Brisbane
    (151.95, -27.56, "qld"),  # Toowoomba
    (139.49, -20.73, "qld"),  # Mount Isa
    (138.60, -34.93, "sa"),  # Adelaide
    (140.78, -37.83, "sa"),  # Mount Gambier
    (134.75, -29.01, "sa"),  # Coober Pedy
    (115.86, -31.95, "wa"),  # Perth
    (122.23, -17.96, "wa"),  # Broome
    (130.84, -12.46, "nt"),  # Darwin
    (133.88, -23.70, "nt"),  # Alice Springs
    (149.13, -35.28, "act"),  # Canberra
    (148.95, -35.60, "act"),  # Namadgi
    (149.24, -35.36, "nsw"),  # Queanbeyan
    (149.07, -35.42, "act"),  # Tuggeranong
    (149.23, -35.42, "nsw"),  # Googong
    (149.44, -35.25, "nsw"),  # Bungendore
    (150.31, -28.547, "qld"),  # Goondiwindi
    (150.36, -28.605, "nsw"),  # Boggabilla, across the Macintyre
    (147.90, -36.20, "vic"),  # Corryong
    (148.13, -36.22, "nsw"),  # Khancoban
    (141.45, -31.95, "nsw"),  # Broken Hill
])
def test_jurisdiction_at(lon, lat, code):
    assert police_car.jurisdiction_at(lon, lat) == code


def test_jurisdiction_for_a_track():
    tas = SimpleNamespace(state="Tasmania")
    mainland = SimpleNamespace(state="")
    sydney, melbourne = (151.21, -33.87), (144.96, -37.81)
    assert police_car.jurisdiction_for(tas, [sydney]) == "tas"  # a region that names its state decides
    assert police_car.jurisdiction_for(SimpleNamespace(state="Victoria")) == "vic"
    assert police_car.jurisdiction_for(mainland, [sydney] * 3) == "nsw"
    # Across a border: the state holding most of the route; a tie goes to the first along it.
    assert police_car.jurisdiction_for(mainland, [melbourne] * 2 + [sydney] * 5) == "nsw"
    assert police_car.jurisdiction_for(mainland, [melbourne, sydney]) == "vic"
    assert police_car.jurisdiction_for(None, np.array([sydney])) == "nsw"
    # With neither a state nor points: the default.
    assert police_car.jurisdiction_for(mainland, []) == police_car.DEFAULT_JURISDICTION
    assert police_car.jurisdiction_for() == police_car.DEFAULT_JURISDICTION
