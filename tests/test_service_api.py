import re
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mapcreator.service.api import create_app
from mapcreator.service.db import STALE_ERROR
from mapcreator.service.routing import NoRoute, RouterUnavailable, check_region
from mapcreator.service.settings import ServiceSettings

TOKEN = "w" * 40
AUTH = {"Authorization": f"Bearer {TOKEN}"}
IP = "203.0.113.7"
MELBOURNE = {"lon": 144.96, "lat": -37.81}  # supported (national data) but a different region from Hobart
AUCKLAND = {"lon": 174.76, "lat": -36.85}  # outside the supported area


class FakeRouter:
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.calls = 0

    def preview(self, waypoints):
        check_region(waypoints)  # as the real routers do
        self.calls += 1
        if self.error:
            raise self.error
        return {"coordinates": [[w.lon, w.lat] for w in waypoints], "length_m": 1000.0}


class FakeTurnstile:
    def verify(self, token: str) -> bool:
        return token == "good"


class FakeStorage:
    def put(self, local_path: Path, key: str) -> None:
        raise AssertionError("the web app never stores files")

    def url(self, key: str) -> str:
        return f"https://downloads.example.org/{key}"

    def delete(self, key: str) -> None:
        pass


def job(lon: float = 147.33, track_id: str = "t", **extra) -> dict:
    return {"track_id": track_id, "waypoints": [{"lon": lon, "lat": -42.81}, {"lon": 147.42, "lat": -42.74}], **extra}


@pytest.fixture
def make(tmp_path):
    """Build an app with test settings, and return a function making clients at given addresses."""
    def build(router=None, turnstile=None, **overrides):
        settings = ServiceSettings(db_path=tmp_path / "jobs.sqlite3", storage_dir=tmp_path / "downloads",
                                   ip_salt="test-salt", **{"worker_token": TOKEN, **overrides})
        app = create_app(settings, router=router or FakeRouter(), turnstile=turnstile, storage=FakeStorage())

        def client(ip: str = IP) -> TestClient:
            return TestClient(app, client=(ip, 50000))

        client.app = app
        return client
    return build


@pytest.fixture
def client(make) -> TestClient:
    return make()()


def submit(c: TestClient, body: dict, **extra):
    return c.post("/api/jobs", json={"job": body, **extra})


def claim(c: TestClient) -> dict:
    r = c.post("/api/worker/claim", headers=AUTH)
    assert r.status_code == 200, r.text
    return r.json()


def test_submit_queues_a_job(client):
    r = submit(client, job())
    assert r.status_code == 201
    assert r.json()["status"] == "queued"
    status = client.get(f"/api/jobs/{r.json()['id']}").json()
    assert status["status"] == "queued" and status["queue_position"] == 1
    assert status["created_at"].endswith("Z") and status["finished_at"] is None
    assert status["download_url"] is None and status["summary"] is None and status["error"] is None


@pytest.mark.parametrize("body", [
    {"job": job(), "dem_url": "http://example.org/x.tif"},
    {"job": job(dem_url="http://example.org/x.tif")},
    {"job": job(track_id="../etc")},
    {"job": {**job(), "waypoints": job()["waypoints"][:1]}},
    {"job": job(width="plus_50")},
    {"job": job(), "turnstile_token": "x" * 3000},
    {"waypoints": job()["waypoints"]},
])
def test_invalid_jobs_get_a_plain_message(client, body):
    r = client.post("/api/jobs", json=body)
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert isinstance(detail, str) and detail.startswith("The request wasn't valid.")
    assert "example.org" not in detail  # submitted values aren't echoed back


def test_out_of_area_names_the_supported_regions(client):
    r = submit(client, {**job(), "waypoints": [job()["waypoints"][0], AUCKLAND]})
    assert r.status_code == 422
    assert r.json()["detail"] == "Point 2 is outside the supported area. Supported regions: Tasmania, Australia."


def test_points_in_different_regions_are_refused(client):
    r = submit(client, {**job(), "waypoints": [job()["waypoints"][0], MELBOURNE]})
    assert r.status_code == 422
    assert r.json()["detail"] == "All points must be in the same region."


def test_queue_cap(make):
    c = make(max_queued_jobs=2)
    assert submit(c("203.0.113.1"), job(147.30)).status_code == 201
    assert submit(c("203.0.113.2"), job(147.31)).status_code == 201
    r = submit(c("203.0.113.3"), job(147.32))
    assert r.status_code == 503 and "queue is full" in r.json()["detail"]
    assert r.headers["Retry-After"]


def test_per_ip_daily_limit(make):
    c = make(jobs_per_ip_per_day=2)
    assert submit(c(), job(147.30)).status_code == 201
    assert submit(c(), job(147.31)).status_code == 201
    r = submit(c(), job(147.32))
    assert r.status_code == 429 and "2 tracks a day" in r.json()["detail"]
    assert submit(c("203.0.113.99"), job(147.32)).status_code == 201


def test_same_road_returns_the_same_job(client):
    first = submit(client, job(track_id="mine"))
    again = submit(client, job(track_id="someone_elses_name"))
    assert again.status_code == 200 and again.json() == first.json()
    # Still the same job once it's built.
    lease = claim(client)
    client.post(f"/api/worker/jobs/{lease['id']}/complete", headers=AUTH,
                json={"worker_id": lease["worker_id"], "result": {"key": f"{lease['id']}/mine.zip"}})
    done = submit(client, job(track_id="later"))
    assert done.status_code == 200 and done.json() == {"id": first.json()["id"], "status": "done"}
    # Different options are a different track.
    assert submit(client, job(width="plus_15")).status_code == 201


def test_a_failed_road_can_be_asked_for_again(client):
    first = submit(client, job()).json()
    lease = claim(client)
    client.post(f"/api/worker/jobs/{lease['id']}/fail", headers=AUTH,
                json={"worker_id": lease["worker_id"], "error": "boom"})
    again = submit(client, job())
    assert again.status_code == 201 and again.json()["id"] != first["id"]


def test_route_preview_and_its_rate_limit(make):
    router = FakeRouter()
    c = make(router=router, previews_per_ip_per_minute=3)
    body = {"waypoints": job()["waypoints"]}
    for _ in range(3):
        r = c().post("/api/route-preview", json=body)
        assert r.status_code == 200 and r.json()["length_m"] == 1000.0
    r = c().post("/api/route-preview", json=body)
    assert r.status_code == 429 and router.calls == 3
    assert c("203.0.113.99").post("/api/route-preview", json=body).status_code == 200


@pytest.mark.parametrize("error, status", [(None, 422), (NoRoute("No drivable route connects these points."), 422),
                                           (RouterUnavailable("Try again in a minute."), 503)])
def test_route_preview_errors(make, error, status):
    c = make(router=FakeRouter(error))()
    points = [job()["waypoints"][0], MELBOURNE] if error is None else job()["waypoints"]
    r = c.post("/api/route-preview", json={"waypoints": points})
    assert r.status_code == status and isinstance(r.json()["detail"], str)


def test_worker_api_does_not_exist_without_a_token(make):
    c = make(worker_token=None)()
    assert c.post("/api/worker/claim", headers=AUTH).status_code == 404
    assert c.post("/api/worker/jobs/x/fail", content=b"{not json",
                  headers={**AUTH, "Content-Type": "application/json"}).status_code == 404


def test_worker_auth(client):
    r = client.post("/api/worker/claim")
    assert r.status_code == 401 and r.headers["WWW-Authenticate"] == "Bearer"
    assert client.post("/api/worker/claim", headers={"Authorization": f"Basic {TOKEN}"}).status_code == 401
    assert client.post("/api/worker/claim", headers={"Authorization": "Bearer " + "x" * 40}).status_code == 403
    # Refused before the body is read, so an outsider never sees a validation error.
    assert client.post("/api/worker/jobs/x/complete", content=b"{not json",
                       headers={"Content-Type": "application/json"}).status_code == 401
    assert client.post("/api/worker/claim", headers=AUTH).status_code == 204


def test_claim_heartbeat_complete(client):
    job_id = submit(client, job(track_id="my_road")).json()["id"]
    lease = claim(client)
    assert lease["id"] == job_id and lease["attempt"] == 1
    # Hosted builds ship under a unique, stable id: mc_<slug>_<first 8 of the route hash>.
    assert re.fullmatch(r"mc_my_road_[0-9a-f]{8}", lease["job"]["track_id"]) and lease["job"]["layouts"] == "both"
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "running"

    url = f"/api/worker/jobs/{job_id}"
    assert client.post(f"{url}/heartbeat", headers=AUTH, json={"worker_id": lease["worker_id"]}).status_code == 200
    assert client.post(f"{url}/heartbeat", headers=AUTH, json={"worker_id": "z" * 22}).status_code == 409

    result = {"key": f"{job_id}/my_road.zip", "summary": {"title": "My Road", "length_m": 1234.5}}
    r = client.post(f"{url}/complete", headers=AUTH, json={"worker_id": lease["worker_id"], "result": result})
    assert r.status_code == 200
    status = client.get(f"/api/jobs/{job_id}").json()
    assert status["status"] == "done" and status["finished_at"].endswith("Z")
    assert status["download_url"] == f"https://downloads.example.org/{job_id}/my_road.zip"
    assert status["summary"] == result["summary"]
    # A second report is refused: the job is no longer running.
    r = client.post(f"{url}/complete", headers=AUTH, json={"worker_id": lease["worker_id"], "result": result})
    assert r.status_code == 409


def test_fail_stores_one_short_line(client):
    job_id = submit(client, job()).json()["id"]
    lease = claim(client)
    error = "Waypoint 2 is 900 m from the nearest drivable road\n" + "x" * 2000
    r = client.post(f"/api/worker/jobs/{job_id}/fail", headers=AUTH,
                    json={"worker_id": lease["worker_id"], "error": error})
    assert r.status_code == 200
    status = client.get(f"/api/jobs/{job_id}").json()
    assert status["status"] == "failed" and status["download_url"] is None
    assert status["error"].startswith("Waypoint 2 is 900 m") and "\n" not in status["error"]
    assert len(status["error"]) <= 500


@pytest.mark.parametrize("key", ["../../etc/passwd", "/abs.zip", "a\\b.zip"])
def test_complete_refuses_unsafe_keys(client, key):
    submit(client, job())
    lease = claim(client)
    r = client.post(f"/api/worker/jobs/{lease['id']}/complete", headers=AUTH,
                    json={"worker_id": lease["worker_id"], "result": {"key": key}})
    assert r.status_code == 422


def test_stale_jobs_are_requeued_on_claim_then_failed(client):
    store = client.app.state.store
    job_id = submit(client, job()).json()["id"]
    first = claim(client)
    store.clock = lambda: time.time() + 11 * 60  # the worker has gone quiet past the 10-minute limit
    second = claim(client)
    assert second["id"] == job_id and second["attempt"] == 2 and second["worker_id"] != first["worker_id"]
    r = client.post(f"/api/worker/jobs/{job_id}/complete", headers=AUTH,
                    json={"worker_id": first["worker_id"], "result": {"key": "a/b.zip"}})
    assert r.status_code == 409  # the first worker lost the job
    store.clock = lambda: time.time() + 30 * 60
    assert client.post("/api/worker/claim", headers=AUTH).status_code == 204
    status = client.get(f"/api/jobs/{job_id}").json()
    assert status["status"] == "failed" and status["error"] == STALE_ERROR


def test_unknown_and_malformed_job_ids(client):
    assert client.get("/api/jobs/" + "a" * 22).status_code == 404
    assert client.get("/api/jobs/not%20an%20id").status_code == 404


def test_security_headers(make):
    c = make()()
    for r in (c.get("/api/health"), c.get("/api/nothing-here"), c.post("/api/worker/claim")):
        csp = r.headers["Content-Security-Policy"]
        assert "script-src 'self';" in csp and "style-src 'self';" in csp
        assert "frame-ancestors 'none'" in csp and "unsafe-inline" not in csp and "unsafe-eval" not in csp
        assert "connect-src 'self' https://tiles.openfreemap.org https://services.thelist.tas.gov.au" in csp
        assert "challenges.cloudflare.com" not in csp
        assert r.headers["X-Content-Type-Options"] == "nosniff"
        assert r.headers["Referrer-Policy"] == "no-referrer"
        assert "geolocation=()" in r.headers["Permissions-Policy"]
        assert all(f"{f}=()" in r.headers["Permissions-Policy"] for f in ("camera", "microphone", "payment"))
        assert r.headers["Cache-Control"] == "no-store"


def test_turnstile(make):
    c = make(turnstile=FakeTurnstile())()
    assert submit(c, job()).status_code == 403
    assert submit(c, job(), turnstile_token="bad").status_code == 403
    assert submit(c, job(), turnstile_token="good").status_code == 201
    csp = c.get("/api/health").headers["Content-Security-Policy"]
    assert "script-src 'self' https://challenges.cloudflare.com;" in csp
    assert "frame-src https://challenges.cloudflare.com;" in csp


def test_client_address_from_a_trusted_proxy_only(make):
    c = make(jobs_per_ip_per_day=1, trusted_proxies=("203.0.113.1",), client_ip_header="X-Forwarded-For")
    proxy = c("203.0.113.1")

    def fwd(ip: str) -> dict:
        return {"X-Forwarded-For": f"192.0.2.250, {ip}"}  # the proxy appends the last entry

    assert proxy.post("/api/jobs", json={"job": job(147.30)}, headers=fwd("198.51.100.1")).status_code == 201
    assert proxy.post("/api/jobs", json={"job": job(147.31)}, headers=fwd("198.51.100.2")).status_code == 201
    assert proxy.post("/api/jobs", json={"job": job(147.32)}, headers=fwd("198.51.100.1")).status_code == 429
    # Anyone else's header is ignored: they're counted by their own address.
    other = c("203.0.113.50")
    assert other.post("/api/jobs", json={"job": job(147.33)}, headers=fwd("198.51.100.3")).status_code == 201
    assert other.post("/api/jobs", json={"job": job(147.34)}, headers=fwd("198.51.100.4")).status_code == 429


def test_a_proxy_missing_from_the_trusted_list_is_named_in_the_log(make, caplog):
    c = make(trusted_proxies=("203.0.113.1",), client_ip_header="CF-Connecting-IP")
    gateway = c("fd00:abcd::1")  # say, a bridge gateway the operator didn't list
    with caplog.at_level("WARNING", logger="mapcreator.service.api"):
        for lon in (147.30, 147.31):
            gateway.post("/api/jobs", json={"job": job(lon)}, headers={"CF-Connecting-IP": "198.51.100.1"})
        c("198.51.100.9").post("/api/jobs", json={"job": job(147.32)},  # a public peer: no warning
                               headers={"CF-Connecting-IP": "198.51.100.1"})
    warnings = [r.getMessage() for r in caplog.records if "TRUSTED_PROXIES" in r.getMessage()]
    assert len(warnings) == 1 and "fd00:abcd::1" in warnings[0]


def test_ipv6_clients_are_counted_per_64(make):
    c = make(jobs_per_ip_per_day=1)
    assert submit(c("2001:db8:1:2::1"), job(147.30)).status_code == 201
    assert submit(c("2001:db8:1:2:ffff::9"), job(147.31)).status_code == 429
    assert submit(c("2001:db8:1:3::1"), job(147.31)).status_code == 201


def test_raw_addresses_are_never_stored(make, tmp_path):
    c = make()
    submit(c("198.51.100.77"), job())
    raw = b"".join(p.read_bytes() for p in tmp_path.glob("jobs.sqlite3*"))
    assert b"198.51.100.77" not in raw


def test_oversized_bodies_are_refused(client):
    r = client.post("/api/jobs", content=b"{" + b" " * 70_000 + b"}", headers={"Content-Type": "application/json"})
    assert r.status_code == 413


def test_health_and_config(make):
    c = make(turnstile=FakeTurnstile(), turnstile_secret="s", turnstile_site_key="site-key")()
    assert c.get("/api/health").json() == {"status": "ok"}
    cfg = c.get("/api/config").json()
    assert cfg["regions"][0]["name"] == "Tasmania" and cfg["max_waypoints"] == 25
    assert cfg["max_length_km"] == 30.0 and cfg["turnstile_site_key"] == "site-key"


def finished_download(tmp_path, **overrides) -> tuple[TestClient, str, bytes]:
    """An app serving its own downloads, with one finished job whose zip is in place."""
    settings = ServiceSettings(db_path=tmp_path / "jobs.sqlite3", storage_dir=tmp_path / "downloads",
                               ip_salt="test-salt", worker_token=TOKEN, **overrides)
    c = TestClient(create_app(settings, router=FakeRouter()), client=(IP, 50000))
    job_id = submit(c, job(track_id="my_road")).json()["id"]
    lease = claim(c)
    data = b"PK\x03\x04" + bytes(range(256)) * 40
    (tmp_path / "downloads" / job_id).mkdir(parents=True)
    (tmp_path / "downloads" / job_id / "my_road.zip").write_bytes(data)
    c.post(f"/api/worker/jobs/{job_id}/complete", headers=AUTH,
           json={"worker_id": lease["worker_id"], "result": {"key": f"{job_id}/my_road.zip"}})
    return c, job_id, data


def test_download_urls_are_absolute(tmp_path):
    c, job_id, _ = finished_download(tmp_path)
    assert c.get(f"/api/jobs/{job_id}").json()["download_url"] == f"http://testserver/downloads/{job_id}/my_road.zip"


def test_download_urls_use_the_public_url_when_set(tmp_path):
    c, job_id, _ = finished_download(tmp_path, public_url="https://mapcreator.example.org/")
    url = c.get(f"/api/jobs/{job_id}").json()["download_url"]
    assert url == f"https://mapcreator.example.org/downloads/{job_id}/my_road.zip"


def test_downloads_are_zip_attachments_content_manager_accepts(tmp_path):
    c, job_id, data = finished_download(tmp_path)
    url = f"/downloads/{job_id}/my_road.zip"
    r = c.get(url)
    assert r.status_code == 200 and r.content == data
    assert r.headers["Content-Type"] == "application/zip"
    assert r.headers["Content-Disposition"] == 'attachment; filename="my_road.zip"'
    assert r.headers["Accept-Ranges"] == "bytes"

    part = c.get(url, headers={"Range": "bytes=4-11"})
    assert part.status_code == 206 and part.content == data[4:12]
    assert part.headers["Content-Range"] == f"bytes 4-11/{len(data)}"

    head = c.head(url)
    assert head.status_code == 200 and head.content == b""
    assert head.headers["Content-Disposition"] == 'attachment; filename="my_road.zip"'


@pytest.mark.parametrize("path", ["/downloads/nothing/here.zip", "/downloads/x/../jobs.sqlite3",
                                  "/downloads/%2e%2e/jobs.sqlite3", "/downloads/a/b.txt", "/downloads/"])
def test_missing_downloads_are_plain_404s_not_pages(tmp_path, path):
    c, _, _ = finished_download(tmp_path)
    r = c.get(path)
    assert r.status_code == 404
    assert "text/html" not in r.headers.get("Content-Type", "")
