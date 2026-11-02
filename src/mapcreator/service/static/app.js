/*
 * mapcreator map page.
 *
 * Runs under a strict content security policy: no inline script or style,
 * so everything is wired up here and styled by class from style.css. Text
 * from the API is only ever set with textContent, never parsed as HTML.
 */
"use strict";

(() => {
  const STYLE_URL = "https://tiles.openfreemap.org/styles/liberty";
  const THELIST_TOPO_TILES =
    "https://services.thelist.tas.gov.au/arcgis/rest/services/Basemaps/Topographic/MapServer/tile/{z}/{y}/{x}";
  const THELIST_TOPO_ATTRIBUTION = "Topographic Basemap from theLIST © State of Tasmania, CC BY 3.0 AU";
  const TASMANIA = [[143.8, -43.7], [148.5, -39.5]];
  const TURNSTILE_SCRIPT = "https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit";
  const PREVIEW_DELAY_MS = 400;
  const POLL_MS = 5000;
  const JOB_HASH = /^#job=([A-Za-z0-9_-]{16,64})$/;
  // What each build stage does, in words (the API reports the stage's name, number and count).
  const STAGE_TEXT = {
    route: "finding the road",
    profile: "measuring the hills",
    section: "shaping the road surface",
    mesh: "building the road and terrain",
    scenery: "placing trees and buildings",
    track: "packaging the track",
    validate: "checking the track",
  };

  const $ = (id) => document.getElementById(id);
  const el = {
    map: $("map"),
    mapError: $("map-error"),
    addCentre: $("add-centre"),
    topo: $("topo-toggle"),
    points: $("points"),
    pointsEmpty: $("points-empty"),
    clear: $("clear-points"),
    routeInfo: $("route-info"),
    form: $("job-form"),
    trackName: $("track-name"),
    trackId: $("track-id"),
    turnstile: $("turnstile"),
    submit: $("submit"),
    job: $("job"),
    jobStatus: $("job-status"),
    builderOffline: $("builder-offline"),
    summary: $("job-summary"),
    download: $("download"),
    downloadLink: $("download-link"),
    contentManagerLink: $("cm-link"),
    error: $("error"),
  };

  const state = {
    config: { max_waypoints: 25, max_length_km: 30, turnstile_site_key: null },
    map: null,
    mapReady: false,
    points: [], // { lng, lat, marker, node }
    route: null, // [[lon, lat], ...] of the latest preview
    routeUsable: false, // the preview found a route within the length limit, or the preview service is down
    previewTimer: null,
    previewSeq: 0,
    previewAbort: null,
    turnstileToken: null,
    turnstileWidget: null,
    submitting: false,
    jobId: null,
    pollTimer: null,
    fallbackId: "track_" + randomSuffix(6),
  };

  // ----- talking to the API -----

  const FALLBACK_MESSAGES = {
    403: "The request was refused. Reload the page and try again.",
    404: "That job doesn't exist any more.",
    413: "The request was too large.",
    422: "Something about that request wasn't valid.",
    429: "Too many requests. Wait a little and try again.",
    503: "The service is busy right now. Try again shortly.",
  };

  class ApiError extends Error {
    constructor(status, message) {
      super(message);
      this.status = status;
    }
  }

  async function api(path, { body, signal } = {}) {
    const init = { method: "GET", headers: { Accept: "application/json" }, signal };
    if (body !== undefined) {
      init.method = "POST";
      init.headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(body);
    }
    let response;
    try {
      response = await fetch(path, init);
    } catch (err) {
      if (err.name === "AbortError") throw err;
      throw new ApiError(0, "Couldn't reach the server. Check your connection and try again.");
    }
    let data = null;
    try {
      data = await response.json();
    } catch (_) {
      data = null;
    }
    if (!response.ok) {
      const detail = data && typeof data.detail === "string" ? data.detail : "";
      const message = detail || FALLBACK_MESSAGES[response.status] || `Something went wrong (error ${response.status}).`;
      throw new ApiError(response.status, message);
    }
    return data;
  }

  // ----- small helpers -----

  function randomSuffix(length) {
    const bytes = new Uint8Array(length);
    crypto.getRandomValues(bytes);
    return Array.from(bytes, (b) => (b % 36).toString(36)).join("");
  }

  // The folder name AC uses: a-z, 0-9 and underscore, at most 32 characters. The server checks it again.
  function slugify(name) {
    return name
      .normalize("NFKD")
      .replace(/[̀-ͯ]/g, "")
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "_")
      .replace(/^_+/, "")
      .slice(0, 32)
      .replace(/_+$/, "");
  }

  function trackId() {
    return slugify(el.trackName.value) || state.fallbackId;
  }

  function round6(x) {
    return Math.round(x * 1e6) / 1e6;
  }

  function formatKm(metres) {
    return (metres / 1000).toLocaleString("en-AU", { maximumFractionDigits: 1, minimumFractionDigits: 1 }) + " km";
  }

  function checkedValue(name) {
    const input = el.form.querySelector(`input[name="${name}"]:checked`);
    return input ? input.value : undefined;
  }

  // The API's download URL as an absolute http(s) URL, or null if it is anything else.
  function downloadUrl(raw) {
    if (typeof raw !== "string") return null;
    let url;
    try {
      url = new URL(raw, window.location.href);
    } catch (_) {
      return null;
    }
    return url.protocol === "https:" || url.protocol === "http:" ? url.href : null;
  }

  // Content Manager downloads the zip from this URL itself, and asks before it installs anything.
  function contentManagerLink(url) {
    return `acmanager://install?url=${encodeURIComponent(url)}`;
  }

  function showError(message) {
    el.error.textContent = message;
    el.error.hidden = false;
  }

  function clearError() {
    el.error.hidden = true;
    el.error.textContent = "";
  }

  // ----- the map -----

  function initMap() {
    if (typeof maplibregl === "undefined") {
      showMapError("The map didn't load. Reload the page to try again.");
      return null;
    }
    let map;
    try {
      map = new maplibregl.Map({
        container: el.map,
        style: STYLE_URL,
        bounds: TASMANIA,
        fitBoundsOptions: { padding: 16 },
        attributionControl: { compact: true },
        dragRotate: false,
        pitchWithRotate: false,
        touchPitch: false,
      });
    } catch (_) {
      showMapError("Your browser couldn't start the map. It needs WebGL switched on.");
      return null;
    }
    map.touchZoomRotate.disableRotation();
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
    map.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-left");
    map.on("load", () => {
      addLayers(map);
      state.mapReady = true;
      drawRoute();
    });
    map.on("click", (event) => {
      // Clicks on (or the end of a drag of) a marker are not new points.
      const target = event.originalEvent && event.originalEvent.target;
      if (target instanceof Element && target.closest(".maplibregl-marker")) return;
      addPoint(event.lngLat);
    });
    return map;
  }

  function showMapError(message) {
    el.mapError.textContent = message;
    el.mapError.hidden = false;
    el.addCentre.disabled = true;
    el.topo.disabled = true;
  }

  function addLayers(map) {
    map.addSource("thelist-topo", {
      type: "raster",
      tiles: [THELIST_TOPO_TILES],
      tileSize: 256,
      maxzoom: 18,
      attribution: THELIST_TOPO_ATTRIBUTION,
    });
    map.addLayer({
      id: "thelist-topo",
      type: "raster",
      source: "thelist-topo",
      layout: { visibility: el.topo.checked ? "visible" : "none" },
    });
    map.addSource("route", { type: "geojson", data: routeGeoJson() });
    const lineLayout = { "line-join": "round", "line-cap": "round" };
    map.addLayer({ id: "route-casing", type: "line", source: "route", layout: lineLayout,
      paint: { "line-color": "#ffffff", "line-width": 8 } });
    map.addLayer({ id: "route-line", type: "line", source: "route", layout: lineLayout,
      paint: { "line-color": "#c2185b", "line-width": 4.5 } });
  }

  function routeGeoJson() {
    const features = state.route && state.route.length > 1
      ? [{ type: "Feature", properties: {}, geometry: { type: "LineString", coordinates: state.route } }]
      : [];
    return { type: "FeatureCollection", features };
  }

  function drawRoute() {
    if (!state.mapReady) return;
    const source = state.map.getSource("route");
    if (source) source.setData(routeGeoJson());
  }

  // ----- route points -----

  function pointRole(i, n) {
    if (i === 0) return "start";
    return i === n - 1 ? "finish" : "via";
  }

  function pointLabel(i, n) {
    const role = pointRole(i, n);
    if (role === "start") return "Start";
    if (role === "finish") return "Finish";
    return `Point ${i + 1}`;
  }

  function addPoint(lngLat) {
    if (state.points.length >= state.config.max_waypoints) {
      showError(`A route can have at most ${state.config.max_waypoints} points.`);
      return;
    }
    clearError();
    const point = { lng: round6(lngLat.lng), lat: round6(lngLat.lat), marker: null, node: null };
    if (state.map) {
      const node = document.createElement("div");
      node.className = "marker";
      const marker = new maplibregl.Marker({ element: node, draggable: true, anchor: "center" })
        .setLngLat([point.lng, point.lat])
        .addTo(state.map);
      marker.on("dragend", () => {
        const at = marker.getLngLat();
        point.lng = round6(at.lng);
        point.lat = round6(at.lat);
        renderPoints();
        schedulePreview();
      });
      point.marker = marker;
      point.node = node;
    }
    state.points.push(point);
    renderPoints();
    schedulePreview();
  }

  function removePoint(index) {
    const [point] = state.points.splice(index, 1);
    if (point && point.marker) point.marker.remove();
    renderPoints();
    schedulePreview();
    // Keep keyboard focus somewhere sensible after the button disappears.
    const buttons = el.points.querySelectorAll("button");
    if (buttons.length) buttons[Math.min(index, buttons.length - 1)].focus();
    else el.addCentre.focus();
  }

  function clearPoints() {
    state.points.forEach((p) => p.marker && p.marker.remove());
    state.points = [];
    renderPoints();
    schedulePreview();
    el.addCentre.focus();
  }

  function renderPoints() {
    const n = state.points.length;
    const items = state.points.map((point, i) => {
      const role = pointRole(i, n);
      const label = pointLabel(i, n);
      if (point.node) {
        // classList, not className: MapLibre keeps its own classes on the marker element.
        point.node.classList.remove("marker-start", "marker-via", "marker-finish");
        point.node.classList.add(`marker-${role}`);
        point.node.textContent = role === "start" ? "S" : role === "finish" ? "F" : String(i + 1);
        point.node.setAttribute("aria-label", `${label} point`);
        point.node.title = `${label}. Drag to move.`;
      }
      const item = document.createElement("li");
      item.className = "point";
      const badge = document.createElement("span");
      badge.className = `point-badge marker-${role}`;
      badge.setAttribute("aria-hidden", "true");
      badge.textContent = role === "start" ? "S" : role === "finish" ? "F" : String(i + 1);
      const name = document.createElement("span");
      name.className = "point-name";
      name.textContent = label;
      const coords = document.createElement("span");
      coords.className = "point-coords";
      coords.textContent = `${point.lat.toFixed(5)}, ${point.lng.toFixed(5)}`;
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "button button-small button-quiet";
      remove.textContent = "Remove";
      remove.setAttribute("aria-label", `Remove the ${label.toLowerCase()} point`);
      remove.addEventListener("click", () => removePoint(i));
      item.append(badge, name, coords, remove);
      return item;
    });
    el.points.replaceChildren(...items);
    el.pointsEmpty.hidden = n > 0;
    el.clear.disabled = n === 0;
  }

  function waypoints() {
    return state.points.map((p) => ({ lon: p.lng, lat: p.lat }));
  }

  // The build routes on its own copy of OpenStreetMap, which can choose a different road from the
  // preview's router. Points sampled along the previewed line, sent as extra waypoints, make the
  // build follow the line the person saw. Only a preview of the current points is used.
  const SHAPE_SPACING_M = 1000;

  function metresBetween(a, b) {
    const k = Math.PI / 180;
    const lat = ((a[1] + b[1]) / 2) * k;
    return 6371000 * Math.hypot((b[0] - a[0]) * k * Math.cos(lat), (b[1] - a[1]) * k);
  }

  function jobWaypoints() {
    const user = waypoints();
    const line = state.routeUsable ? state.route : null;
    if (!line || line.length < 2 || user.length < 2) return user;
    const cum = [0];
    for (let i = 1; i < line.length; i++) cum.push(cum[i - 1] + metresBetween(line[i - 1], line[i]));
    const total = cum[cum.length - 1];
    const count = Math.min(state.config.max_waypoints - 2, Math.floor(total / SHAPE_SPACING_M));
    if (count < 1) return user;
    const out = [user[0]];
    let j = 1;
    for (let k = 1; k <= count; k++) {
      const target = (total * k) / (count + 1);
      while (j < line.length - 1 && cum[j] < target) j++;
      const a = line[j - 1];
      const b = line[j];
      const f = (target - cum[j - 1]) / (cum[j] - cum[j - 1] || 1);
      out.push({ lon: round6(a[0] + (b[0] - a[0]) * f), lat: round6(a[1] + (b[1] - a[1]) * f) });
    }
    out.push(user[user.length - 1]);
    return out;
  }

  // ----- route preview -----

  function schedulePreview() {
    clearTimeout(state.previewTimer);
    if (state.previewAbort) state.previewAbort.abort();
    state.previewSeq += 1;
    state.routeUsable = false;
    if (state.points.length < 2) {
      state.route = null;
      drawRoute();
      el.routeInfo.textContent = state.points.length === 1 ? "Now add a finish point." : "";
      updateSubmit();
      return;
    }
    el.routeInfo.textContent = "Finding the route…";
    updateSubmit();
    state.previewTimer = setTimeout(runPreview, PREVIEW_DELAY_MS);
  }

  async function runPreview() {
    const seq = state.previewSeq;
    const abort = new AbortController();
    state.previewAbort = abort;
    try {
      const preview = await api("/api/route-preview", { body: { waypoints: waypoints() }, signal: abort.signal });
      if (seq !== state.previewSeq) return;
      state.route = preview.coordinates;
      const limitM = state.config.max_length_km * 1000;
      if (preview.length_m > limitM) {
        el.routeInfo.textContent = `The route is ${formatKm(preview.length_m)}, longer than the ` +
          `${state.config.max_length_km} km limit. Move the points closer together.`;
      } else {
        el.routeInfo.textContent = `Route length: ${formatKm(preview.length_m)}. ` +
          "The build checks the route again against the latest road data.";
        state.routeUsable = true;
      }
      clearError();
    } catch (err) {
      if (err.name === "AbortError" || seq !== state.previewSeq) return;
      state.route = null;
      if (err.status === 0 || err.status === 503) {
        // The preview service is down, not the road: let people build anyway.
        el.routeInfo.textContent = "The route preview isn't available right now. You can still build the track.";
        state.routeUsable = true;
      } else {
        el.routeInfo.textContent = "";
        showError(err.message);
      }
    }
    drawRoute();
    updateSubmit();
  }

  // ----- submitting and tracking a job -----

  function updateSubmit() {
    const needsChallenge = Boolean(state.config.turnstile_site_key);
    el.submit.disabled = state.submitting || state.points.length < 2 || !state.routeUsable ||
      (needsChallenge && !state.turnstileToken);
  }

  function updateTrackId() {
    el.trackId.textContent = trackId();
  }

  async function submitJob(event) {
    event.preventDefault();
    if (el.submit.disabled) return;
    clearError();
    const body = {
      job: {
        track_id: trackId(),
        waypoints: jobWaypoints(),
        width: checkedValue("width"),
        layouts: checkedValue("layouts"),
      },
    };
    if (state.config.turnstile_site_key) body.turnstile_token = state.turnstileToken;
    state.submitting = true;
    el.submit.textContent = "Sending…";
    updateSubmit();
    try {
      const job = await api("/api/jobs", { body });
      trackJob(job.id);
    } catch (err) {
      showError(err.message);
    } finally {
      state.submitting = false;
      el.submit.textContent = "Build my track";
      resetTurnstile();
      updateSubmit();
    }
  }

  function trackJob(id) {
    state.jobId = id;
    clearTimeout(state.pollTimer);
    try {
      history.replaceState(null, "", `#job=${id}`);
    } catch (_) {
      /* a bookmarkable link is a nicety */
    }
    el.job.hidden = false;
    el.download.hidden = true;
    el.summary.hidden = true;
    el.jobStatus.textContent = "Checking on your track…";
    pollJob();
  }

  async function pollJob() {
    const id = state.jobId;
    try {
      const job = await api(`/api/jobs/${encodeURIComponent(id)}`);
      if (id !== state.jobId) return;
      showJob(job);
      if (job.status === "queued" || job.status === "running") state.pollTimer = setTimeout(pollJob, POLL_MS);
    } catch (err) {
      if (id !== state.jobId) return;
      if (err.status === 404) {
        el.job.hidden = true;
        showError(err.message);
        return;
      }
      el.jobStatus.textContent = "Couldn't check on the build just now. Trying again…";
      state.pollTimer = setTimeout(pollJob, POLL_MS * 2);
    }
  }

  function showJob(job) {
    const offline = job.builder_online === false
      ? " The track builder is offline right now, so it will start when the builder is back."
      : "";
    const stage = job.stage && STAGE_TEXT[job.stage.name]
      ? ` Step ${job.stage.number} of ${job.stage.of}: ${STAGE_TEXT[job.stage.name]}.`
      : " This usually takes a few minutes.";
    const messages = {
      queued: (job.queue_position
        ? `Waiting in the queue: number ${job.queue_position}.`
        : "Waiting in the queue.") + offline,
      running: `Building your track.${stage}${offline}`,
      done: "Your track is ready.",
      failed: `The build failed. ${job.error || ""}`.trim(),
    };
    el.jobStatus.textContent = messages[job.status] || "Unknown status.";
    const url = job.status === "done" ? downloadUrl(job.download_url) : null;
    if (url) {
      el.downloadLink.href = url;
      el.contentManagerLink.href = contentManagerLink(url);
      el.download.hidden = false;
      showSummary(job.summary || {});
    }
  }

  // Warn before anyone submits if no builder has been heard from lately. Nothing to say if this fails.
  async function checkBuilder() {
    try {
      const status = await api("/api/status");
      el.builderOffline.hidden = status.builder_online !== false;
    } catch (_) {
      el.builderOffline.hidden = true;
    }
  }

  function showSummary(summary) {
    const rows = [];
    if (typeof summary.title === "string") rows.push(["Road", summary.title]);
    if (typeof summary.length_m === "number") rows.push(["Length", formatKm(summary.length_m)]);
    if (Array.isArray(summary.layouts)) {
      const names = summary.layouts.map((l) => l && l.name).filter((name) => typeof name === "string");
      if (names.length) rows.push(["Layouts", names.join("; ")]);
    }
    if (typeof summary.zip_mb === "number") rows.push(["Download size", `${summary.zip_mb} MB`]);
    el.summary.replaceChildren(...rows.flatMap(([term, value]) => {
      const dt = document.createElement("dt");
      dt.textContent = term;
      const dd = document.createElement("dd");
      dd.textContent = value;
      return [dt, dd];
    }));
    el.summary.hidden = rows.length === 0;
  }

  // ----- Cloudflare Turnstile (only when the server has it switched on) -----

  function loadTurnstile(siteKey) {
    el.turnstile.hidden = false;
    const script = document.createElement("script");
    script.src = TURNSTILE_SCRIPT;
    script.async = true;
    script.addEventListener("load", () => {
      if (!window.turnstile) return;
      state.turnstileWidget = window.turnstile.render(el.turnstile, {
        sitekey: siteKey,
        callback: (token) => {
          state.turnstileToken = token;
          updateSubmit();
        },
        "expired-callback": () => {
          state.turnstileToken = null;
          updateSubmit();
        },
        "error-callback": () => {
          state.turnstileToken = null;
          updateSubmit();
        },
      });
    });
    script.addEventListener("error", () => {
      showError("The anti-robot check couldn't load. Reload the page to try again.");
    });
    document.head.appendChild(script);
  }

  function resetTurnstile() {
    if (state.turnstileWidget !== null && window.turnstile) {
      state.turnstileToken = null;
      window.turnstile.reset(state.turnstileWidget);
    }
  }

  // ----- start -----

  async function init() {
    el.addCentre.addEventListener("click", () => {
      if (state.map) addPoint(state.map.getCenter());
    });
    el.topo.addEventListener("change", () => {
      if (state.mapReady) {
        state.map.setLayoutProperty("thelist-topo", "visibility", el.topo.checked ? "visible" : "none");
      }
    });
    el.clear.addEventListener("click", clearPoints);
    el.trackName.addEventListener("input", updateTrackId);
    el.form.addEventListener("submit", submitJob);

    updateTrackId();
    renderPoints();
    updateSubmit();
    state.map = initMap();

    try {
      Object.assign(state.config, await api("/api/config"));
    } catch (err) {
      showError(err.message);
    }
    if (state.config.turnstile_site_key) loadTurnstile(state.config.turnstile_site_key);
    updateSubmit();
    checkBuilder();

    const resumed = JOB_HASH.exec(window.location.hash);
    if (resumed) trackJob(resumed[1]);
  }

  init();
})();
