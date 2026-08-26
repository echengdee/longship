from __future__ import annotations

import json
import queue
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from collections.abc import Iterable, Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, TextIO

from longship.runtime.follow_person import FollowEventSink


class CompositeEventSink:
    def __init__(self, sinks: Iterable[FollowEventSink]) -> None:
        self._sinks = tuple(sinks)

    def publish(self, event: Mapping[str, Any]) -> None:
        for sink in self._sinks:
            sink.publish(event)


class BufferedEventSink:
    """Keep journal/dashboard latency and failure out of the control caller."""

    def __init__(
        self,
        downstream: FollowEventSink,
        *,
        maximum_pending_events: int = 512,
    ) -> None:
        if not 8 <= maximum_pending_events <= 10_000:
            raise ValueError("event buffer size is outside supported bounds")
        self._downstream = downstream
        self._queue: queue.Queue[Mapping[str, Any] | None] = queue.Queue(
            maxsize=maximum_pending_events
        )
        self._thread: threading.Thread | None = None
        self.dropped_events = 0
        self.last_error: str | None = None

    def start(self) -> "BufferedEventSink":
        if self._thread is not None:
            raise RuntimeError("buffered event sink is already running")
        self._thread = threading.Thread(
            target=self._run,
            name="longship-follow-events",
            daemon=True,
        )
        self._thread.start()
        return self

    def publish(self, event: Mapping[str, Any]) -> None:
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                pass
            self.dropped_events += 1
            try:
                self._queue.put_nowait(event)
            except queue.Full:
                self.dropped_events += 1

    def close(self) -> None:
        thread = self._thread
        if thread is None:
            return
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                pass
            self._queue.put_nowait(None)
        thread.join(timeout=2.0)
        self._thread = None

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                try:
                    self._downstream.publish(item)
                except Exception as exc:
                    self.last_error = type(exc).__name__
            finally:
                self._queue.task_done()

    def __enter__(self) -> "BufferedEventSink":
        return self.start()

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        self.close()


class JsonlEventSink:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._stream: TextIO | None = None
        self._lock = threading.Lock()

    def open(self) -> "JsonlEventSink":
        if self._stream is not None:
            raise RuntimeError("event journal is already open")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("x", encoding="utf-8", buffering=1)
        return self

    def publish(self, event: Mapping[str, Any]) -> None:
        if self._stream is None:
            raise RuntimeError("event journal is not open")
        serialized = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            self._stream.write(serialized + "\n")

    def close(self) -> None:
        if self._stream is None:
            return
        with self._lock:
            self._stream.close()
            self._stream = None

    def __enter__(self) -> "JsonlEventSink":
        return self.open()

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        self.close()


class FollowDashboard:
    """Read-only live state view; never participates in the control path."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8093,
        history_limit: int = 120,
        camera_preview_url: str | None = None,
    ) -> None:
        if not isinstance(host, str) or not host:
            raise ValueError("dashboard host is required")
        if type(port) is not int or not 0 <= port <= 65_535:
            raise ValueError("dashboard port is invalid")
        if not 10 <= history_limit <= 2_000:
            raise ValueError("dashboard history limit is invalid")
        self.host = host
        self.port = port
        self._camera_preview_url = self._validate_camera_url(camera_preview_url)
        self._lock = threading.Lock()
        self._latest: dict[str, Any] | None = None
        self._latest_system: dict[str, Any] | None = None
        self._latest_scene: dict[str, Any] | None = None
        self._history: deque[dict[str, Any]] = deque(maxlen=history_limit)
        self._camera_frame: tuple[int, bytes, str] | None = None
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def publish(self, event: Mapping[str, Any]) -> None:
        copied = dict(event)
        schema = copied.get("schema_version")
        if schema == "longship.follow-system-event.v0":
            with self._lock:
                self._latest_system = copied
            return
        if schema != "longship.follow-runtime-event.v1":
            return
        snapshot = copied.get("snapshot")
        if isinstance(snapshot, Mapping):
            history_item = {
                "revision": snapshot.get("revision"),
                "state": snapshot.get("state"),
                "detail": snapshot.get("detail"),
            }
        else:
            history_item = {"revision": None, "state": "invalid", "detail": ""}
        with self._lock:
            self._latest = copied
            scene = copied.get("scene")
            if isinstance(scene, Mapping):
                self._latest_scene = dict(scene)
            previous = self._history[-1] if self._history else None
            if previous is None or (
                previous.get("state"), previous.get("detail")
            ) != (history_item["state"], history_item["detail"]):
                self._history.append(history_item)

    def publish_camera_frame(
        self, sequence: int, jpeg: bytes, *, source: str
    ) -> None:
        if type(sequence) is not int or sequence < 0:
            raise ValueError("camera sequence must be a non-negative integer")
        if not isinstance(jpeg, bytes) or not 4 <= len(jpeg) <= 5_000_000:
            raise ValueError("camera JPEG size is outside supported bounds")
        if not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9"):
            raise ValueError("camera frame is not a complete JPEG")
        if not isinstance(source, str) or not source.strip() or len(source) > 100:
            raise ValueError("camera source label is invalid")
        with self._lock:
            self._camera_frame = (sequence, bytes(jpeg), source)

    def start(self) -> "FollowDashboard":
        if self._server is not None:
            raise RuntimeError("dashboard is already running")
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                route = self.path.split("?", 1)[0]
                if route == "/":
                    self._send(
                        HTTPStatus.OK,
                        "text/html; charset=utf-8",
                        _DASHBOARD_HTML,
                    )
                    return
                if route == "/api/snapshot":
                    with owner._lock:
                        camera = owner._camera_frame
                        body = json.dumps(
                            {
                                "event": owner._latest,
                                "system_event": owner._latest_system,
                                "last_scene": owner._latest_scene,
                                "history": list(owner._history),
                                "camera": {
                                    "available": bool(
                                        camera or owner._camera_preview_url
                                    ),
                                    "sequence": camera[0] if camera else None,
                                    "source": (
                                        camera[2]
                                        if camera
                                        else (
                                            "external-rgbd"
                                            if owner._camera_preview_url
                                            else None
                                        )
                                    ),
                                },
                                "server_time_unix_ms": time.time_ns() // 1_000_000,
                            },
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    self._send(HTTPStatus.OK, "application/json", body)
                    return
                if route == "/camera.jpg":
                    self._camera()
                    return
                self._send(HTTPStatus.NOT_FOUND, "text/plain", b"not found\n")

            def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                self._send(
                    HTTPStatus.METHOD_NOT_ALLOWED,
                    "text/plain",
                    b"dashboard is read-only\n",
                )

            def log_message(self, format: str, *args: object) -> None:
                del format, args

            def _send(
                self, status: HTTPStatus, content_type: str, body: str | bytes
            ) -> None:
                encoded = body.encode("utf-8") if isinstance(body, str) else body
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(encoded)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'self'; img-src 'self' data:; "
                    "style-src 'unsafe-inline'; script-src 'unsafe-inline'",
                )
                self.end_headers()
                self.wfile.write(encoded)

            def _camera(self) -> None:
                with owner._lock:
                    cached = owner._camera_frame
                if cached is not None:
                    self._send(HTTPStatus.OK, "image/jpeg", cached[1])
                    return
                if owner._camera_preview_url is None:
                    self._send(
                        HTTPStatus.NOT_FOUND,
                        "text/plain",
                        b"camera preview is unavailable\n",
                    )
                    return
                try:
                    request = urllib.request.Request(
                        owner._camera_preview_url,
                        headers={
                            "Accept": "image/jpeg",
                            "User-Agent": "longship-follow-hud/0",
                        },
                    )
                    with urllib.request.urlopen(request, timeout=0.3) as response:
                        final = urllib.parse.urlparse(response.geturl())
                        expected = urllib.parse.urlparse(owner._camera_preview_url)
                        if (
                            (final.hostname or "").lower()
                            != (expected.hostname or "").lower()
                            or (final.port or 80) != (expected.port or 80)
                        ):
                            raise ValueError("camera preview redirected")
                        if response.headers.get_content_type() != "image/jpeg":
                            raise ValueError("camera preview is not JPEG")
                        frame = response.read(5_000_001)
                    if len(frame) > 5_000_000:
                        raise ValueError("camera preview exceeds size limit")
                    if not frame.startswith(b"\xff\xd8") or not frame.endswith(
                        b"\xff\xd9"
                    ):
                        raise ValueError("camera preview is malformed")
                except (
                    OSError,
                    TimeoutError,
                    urllib.error.URLError,
                    ValueError,
                ):
                    self._send(
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        "text/plain",
                        b"camera preview fetch failed\n",
                    )
                    return
                self._send(HTTPStatus.OK, "image/jpeg", frame)

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="longship-follow-dashboard",
            daemon=True,
        )
        self._thread.start()
        return self

    def close(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._server = None
        self._thread = None

    def __enter__(self) -> "FollowDashboard":
        return self.start()

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        self.close()

    @staticmethod
    def _validate_camera_url(value: str | None) -> str | None:
        if value is None:
            return None
        parsed = urllib.parse.urlparse(value)
        if parsed.scheme != "http" or not parsed.hostname:
            raise ValueError("camera preview must be an HTTP URL")
        if parsed.hostname.lower() not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("camera preview must remain on loopback")
        carries_credentials = bool(parsed.username or parsed.password)
        if carries_credentials or parsed.query or parsed.fragment:
            raise ValueError("camera preview URL contains forbidden components")
        return value


_DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Longship Mission HUD</title>
<style>
:root{color-scheme:dark;--bg:#071018;--panel:#0c1823e8;--edge:#22394b;--text:#e9f2f7;--muted:#849aaa;--cyan:#50d8e8;--green:#67efb2;--amber:#ffb95c;--red:#ff6073;--violet:#b998ff;font:14px Inter,ui-sans-serif,system-ui,sans-serif}
*{box-sizing:border-box}body{margin:0;min-height:100vh;color:var(--text);background:radial-gradient(circle at 42% -20%,#16364a 0,#071018 48%);padding:18px}
header{height:64px;display:flex;align-items:center;justify-content:space-between;border:1px solid var(--edge);border-radius:13px;background:#0b1721d9;padding:0 18px;box-shadow:0 18px 50px #0007}
.brand{display:flex;align-items:center;gap:12px}.mark{width:34px;height:34px;border:1px solid #3d6b80;border-radius:9px;display:grid;place-items:center;color:var(--cyan);font-weight:800;background:linear-gradient(145deg,#153448,#0a1720)}
h1{font-size:17px;letter-spacing:.08em;margin:0}.sub{color:var(--muted);font-size:11px;letter-spacing:.14em;margin-top:3px;text-transform:uppercase}
.live{display:flex;align-items:center;gap:8px;color:#b8c8d1;font:12px ui-monospace,monospace}.live i{width:8px;height:8px;border-radius:50%;background:#49606e;box-shadow:0 0 0 4px #263c4844}.live.online i{background:var(--green);box-shadow:0 0 13px var(--green)}
.layout{margin-top:14px;display:grid;grid-template-columns:minmax(360px,1.25fr) minmax(390px,1fr) 310px;grid-template-rows:auto auto;gap:14px}.card{position:relative;border:1px solid var(--edge);border-radius:13px;background:var(--panel);box-shadow:0 14px 36px #0005;overflow:hidden}.head{height:43px;padding:0 14px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #1d3241;color:#bdd0da;font-size:12px;letter-spacing:.1em;text-transform:uppercase}.head small{color:#6e8797;font:10px ui-monospace,monospace;letter-spacing:.04em}
.camera{min-height:390px;background:#050b10}.camera img{display:block;width:100%;height:100%;min-height:390px;max-height:520px;object-fit:cover;background:linear-gradient(135deg,#07131c,#101e29)}.empty{position:absolute;inset:44px 0 0;display:grid;place-items:center;color:#526b7a;letter-spacing:.08em}.empty[hidden]{display:none}.overlay{position:absolute;left:12px;bottom:12px;display:flex;gap:7px}.chip{border:1px solid #315063;background:#08131bdc;border-radius:6px;padding:5px 8px;color:#a8bfcb;font:10px ui-monospace,monospace;text-transform:uppercase}
.mapwrap{padding:10px;background:#08131c}.mapwrap canvas{display:block;width:100%;height:auto;border-radius:8px;background:#061019}.rail{grid-row:1 / span 2;grid-column:3;padding-bottom:12px}.section{padding:11px 14px 4px;color:#6f8796;font-size:10px;letter-spacing:.14em;text-transform:uppercase}.metric{min-height:36px;margin:0 14px;display:grid;grid-template-columns:1fr auto;align-items:center;border-bottom:1px solid #172b38;color:#91a7b4}.metric b{font:12px ui-monospace,monospace;color:#d6e5eb;font-weight:500}.metric b.good{color:var(--green)}.metric b.warn{color:var(--amber)}.metric b.fail{color:var(--red)}
.mission{grid-column:1 / span 2;display:grid;grid-template-columns:1.3fr 1fr;min-height:160px}.detail{padding:15px 16px;border-right:1px solid #1d3241}.stateLine{display:flex;gap:9px;align-items:center}.stateBadge{padding:5px 9px;border-radius:6px;background:#173044;color:var(--cyan);font:11px ui-monospace,monospace;text-transform:uppercase}.detail p{color:#a6b8c2;line-height:1.5;margin:12px 0 0}.legend{display:flex;gap:13px;flex-wrap:wrap;margin-top:15px;color:#718897;font-size:11px}.legend span:before{content:'';display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;background:var(--dot)}
.history{padding:11px 14px}.history ol{list-style:none;margin:0;padding:0;max-height:125px;overflow:auto}.history li{padding:7px 0;border-bottom:1px solid #172b38;color:#849aa8;font-size:11px}.history li strong{color:#b9cbd4;font:10px ui-monospace,monospace;margin-right:7px;text-transform:uppercase}
@media(max-width:1180px){.layout{grid-template-columns:1fr 1fr}.rail{grid-column:1 / span 2;grid-row:auto;display:grid;grid-template-columns:repeat(3,1fr)}.rail .head,.rail .section{grid-column:1 / -1}.mission{grid-column:1 / span 2}}
@media(max-width:760px){body{padding:8px}.layout{grid-template-columns:1fr}.camera,.map,.rail,.mission{grid-column:1;grid-row:auto}.rail{display:block}.mission{display:block}.detail{border-right:0;border-bottom:1px solid #1d3241}header{height:auto;padding:12px}.camera img{min-height:260px}}
</style>
</head>
<body>
<header><div class="brand"><div class="mark">LS</div><div><h1>LONGSHIP MISSION HUD</h1><div class="sub">FollowPerson · read-only observability</div></div></div><div id="live" class="live"><i></i><span id="updated">waiting for telemetry</span></div></header>
<main class="layout">
<section class="card camera"><div class="head"><span>Perception view</span><small id="cameraLabel">NO STREAM</small></div><img id="camera" alt="robot perception preview" hidden><div id="cameraEmpty" class="empty">CAMERA STREAM UNAVAILABLE</div><div class="overlay"><span class="chip" id="cameraSeq">FRAME —</span><span class="chip" id="cameraTrack">TARGET —</span></div></section>
<section class="card map"><div class="head"><span>Local environment</span><small>BASE FRAME · METRES</small></div><div class="mapwrap"><canvas id="map" width="760" height="560"></canvas></div></section>
<aside class="card rail"><div class="head"><span>Telemetry</span><small id="provider">PROVIDER —</small></div><div class="section">Mission</div>
<div class="metric"><span>Runtime</span><b id="state">WAITING</b></div><div class="metric"><span>Brain event</span><b id="brain">—</b></div><div class="metric"><span>Agent</span><b id="agent">DETERMINISTIC</b></div><div class="metric"><span>Skill call</span><b id="skill">—</b></div><div class="metric"><span>Task graph</span><b id="graph">—</b></div><div class="metric"><span>Task node</span><b id="node">—</b></div><div class="metric"><span>Scene</span><b id="seq">—</b></div><div class="metric"><span>Track</span><b id="track">—</b></div>
<div class="section">Target geometry</div><div class="metric"><span>Target · base</span><b id="target">—</b></div><div class="metric"><span>Next goal · base</span><b id="goal">—</b></div><div class="metric"><span>Target · world</span><b id="targetWorld">—</b></div><div class="metric"><span>Follow goal · world</span><b id="goalWorld">—</b></div>
<div class="section">Motion & safety</div><div class="metric"><span>Command</span><b id="command">0.000 / 0.000</b></div><div class="metric"><span>Clearance</span><b id="clearance">—</b></div><div class="metric"><span>Robot · world</span><b id="robotWorld">—</b></div><div class="metric"><span>Base height</span><b id="height">—</b></div><div class="metric"><span>Tilt</span><b id="tilt">—</b></div><div class="metric"><span>Contacts</span><b id="contacts">—</b></div><div class="metric"><span>Policy / physics</span><b id="steps">—</b></div></aside>
<section class="card mission"><div class="detail"><div class="stateLine"><span id="stateBadge" class="stateBadge">WAITING</span><span id="missionId" class="sub">NO ACTIVE SESSION</span></div><p id="detail">No Runtime event has arrived.</p><div class="legend"><span style="--dot:#67aefc">robot</span><span style="--dot:#67efb2">locked person</span><span style="--dot:#b998ff">next goal</span><span style="--dot:#ffb95c">obstacle</span><span style="--dot:#50d8e8">planned path</span></div></div><div class="history"><div class="section">State transitions</div><ol id="history"></ol></div></section>
</main>
<script>
const $=id=>document.getElementById(id),canvas=$('map'),ctx=canvas.getContext('2d');
const scale=102,origin={x:canvas.width/2,y:canvas.height-47};let lastCamera='';
const fmt=(v,d=2)=>Number.isFinite(v)?Number(v).toFixed(d):'—';
const point=(forward,left)=>({x:origin.x-left*scale,y:origin.y-forward*scale});
function dot(forward,left,radius,color){const p=point(forward,left);ctx.beginPath();ctx.arc(p.x,p.y,Math.max(4,radius*scale),0,Math.PI*2);ctx.fillStyle=color;ctx.fill()}
function label(text,p,color){ctx.font='11px ui-monospace,monospace';ctx.fillStyle=color;ctx.fillText(text,p.x+9,p.y-9)}
function cross(forward,left,color){const p=point(forward,left);ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(p.x-9,p.y);ctx.lineTo(p.x+9,p.y);ctx.moveTo(p.x,p.y-9);ctx.lineTo(p.x,p.y+9);ctx.stroke();ctx.strokeRect(p.x-4,p.y-4,8,8);return p}
function drawMap(event){ctx.clearRect(0,0,canvas.width,canvas.height);ctx.fillStyle='#061019';ctx.fillRect(0,0,canvas.width,canvas.height);ctx.font='10px ui-monospace,monospace';
 for(let f=0;f<=5;f++){const p=point(f,0);ctx.strokeStyle=f===0?'#31586e':'#152c39';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(18,p.y);ctx.lineTo(canvas.width-18,p.y);ctx.stroke();ctx.fillStyle='#506d7d';ctx.fillText(f+'m',22,p.y-5)}
 for(let l=-3;l<=3;l++){const p=point(0,l);ctx.strokeStyle=l===0?'#31586e':'#102632';ctx.beginPath();ctx.moveTo(p.x,18);ctx.lineTo(p.x,canvas.height-22);ctx.stroke();if(l)ctx.fillText((l>0?'+':'')+l+'L',p.x+4,canvas.height-27)}
 const top=point(4.7,0),left=point(0,2.1),right=point(0,-2.1);ctx.fillStyle='#17364722';ctx.beginPath();ctx.moveTo(origin.x,origin.y);ctx.lineTo(left.x,left.y);ctx.lineTo(top.x,top.y);ctx.lineTo(right.x,right.y);ctx.closePath();ctx.fill();
 const s=(event&&event.snapshot)||{},scene=(event&&event.scene)||{};(scene.obstacles||[]).forEach(o=>{dot(o.forward_m,o.left_m,o.radius_m,'#ffb95c88');label('OBS',point(o.forward_m,o.left_m),'#e7a94f')});
 const path=s.path_robot_xy_m||[];if(path.length){ctx.strokeStyle='#50d8e8';ctx.lineWidth=3;ctx.beginPath();path.forEach((v,i)=>{const p=point(v[0],v[1]);i?ctx.lineTo(p.x,p.y):ctx.moveTo(p.x,p.y)});ctx.stroke();const g=path[path.length-1],gp=cross(g[0],g[1],'#b998ff');label('NEXT '+fmt(g[0])+','+fmt(g[1]),gp,'#c7afff')}
 (scene.tracks||[]).forEach(t=>{const locked=t.track_id===s.locked_track_id;dot(t.forward_m,t.left_m,locked ? .13 : .09,locked?'#67efb2':'#8298a4');if(locked)label('PERSON '+fmt(t.forward_m)+','+fmt(t.left_m),point(t.forward_m,t.left_m),'#75efb8')});
 if(Array.isArray(s.target_robot_xy_m)){const t=s.target_robot_xy_m,p=cross(t[0],t[1],'#67efb2');label('TARGET',p,'#67efb2')}
 ctx.save();ctx.translate(origin.x,origin.y);ctx.fillStyle='#67aefc';ctx.beginPath();ctx.moveTo(0,-15);ctx.lineTo(-11,11);ctx.lineTo(11,11);ctx.closePath();ctx.fill();ctx.restore();label('G1 / BASE',origin,'#76baff')}
function classify(state){return state==='failed'||state==='stop_unverified'?'fail':state==='blocked'||state==='holding_for_scene'?'warn':'good'}
function coordinate(value){return Array.isArray(value)?fmt(value[0])+', '+fmt(value[1]):'—'}
function render(payload){const e=payload.event||{},s=e.snapshot||{},scene=e.scene||payload.last_scene||{},displayEvent={...e,scene},sys=payload.system_event||{},graph=sys.task_graph||{},tele=e.target_telemetry||{},camera=payload.camera||{};drawMap(displayEvent);
 const state=s.state||'waiting',stateClass=classify(state);$('state').textContent=state.toUpperCase();$('state').className=stateClass;$('stateBadge').textContent=state.toUpperCase();$('stateBadge').className='stateBadge '+stateClass;$('seq').textContent=scene.sequence??'—';$('track').textContent=s.locked_track_id||'—';$('cameraTrack').textContent='TARGET '+(s.locked_track_id||'—');$('missionId').textContent=s.session_id||'NO ACTIVE SESSION';$('detail').textContent=s.detail||'No Runtime event has arrived.';
 $('brain').textContent=sys.event_type||'—';$('agent').textContent=tele.brain_provider==='codex'?((tele.brain_model||'CODEX')+' · '+(tele.brain_reasoning_effort||'default')).toUpperCase():(tele.brain_provider||'deterministic').toUpperCase();$('skill').textContent=sys.active_skill_call_id||'—';$('graph').textContent=graph.graph_id?graph.graph_id+' · '+(graph.state||'—'):'—';$('node').textContent=graph.current_operation_id||'—';$('target').textContent=coordinate(s.target_robot_xy_m);const path=s.path_robot_xy_m||[];$('goal').textContent=coordinate(path.length?path[path.length-1]:null);$('targetWorld').textContent=coordinate(tele.person_world_xy_m);$('goalWorld').textContent=coordinate(tele.follow_goal_world_xy_m);const cmd=s.command||{};$('command').textContent=fmt(cmd.forward_mps||0,3)+' / '+fmt(cmd.yaw_rate_radps||0,3);$('clearance').textContent=scene.raw_forward_clearance_m==null?'—':fmt(scene.raw_forward_clearance_m)+' m';
 const provider=tele.provider||'provider —';$('provider').textContent=provider==='external-unitree-rl-gym-g1-12dof'?'UNITREE G1 · 12DOF':provider.toUpperCase();$('provider').title=provider;$('robotWorld').textContent=Array.isArray(tele.robot_world_pose)?fmt(tele.robot_world_pose[0])+', '+fmt(tele.robot_world_pose[1])+', '+fmt(tele.robot_world_pose[2]):'—';$('height').textContent=Number.isFinite(tele.base_height_m)?fmt(tele.base_height_m,3)+' m':'—';$('tilt').textContent=Number.isFinite(tele.tilt_rad)?fmt(tele.tilt_rad,3)+' rad':'—';$('tilt').className=tele.fallen?'fail':'good';$('contacts').textContent=Number.isFinite(tele.barrier_contact_steps)?String(tele.barrier_contact_steps):'—';$('contacts').className=tele.barrier_contact_steps>0?'fail':'good';$('steps').textContent=Number.isFinite(tele.policy_steps)?tele.policy_steps+' / '+tele.physics_steps:'—';
 const key=(camera.source||'none')+':'+(camera.sequence??scene.sequence??'none');if(camera.available){$('camera').hidden=false;$('cameraEmpty').hidden=true;if(key!==lastCamera){$('camera').src='/camera.jpg?v='+encodeURIComponent(key);lastCamera=key}$('cameraLabel').textContent=(camera.source||'camera').toUpperCase();$('cameraSeq').textContent='FRAME '+(camera.sequence??scene.sequence??'—')}else{$('camera').hidden=true;$('cameraEmpty').hidden=false;$('cameraLabel').textContent='NO STREAM'}
 const history=$('history');history.replaceChildren();(payload.history||[]).slice(-9).reverse().forEach(h=>{const li=document.createElement('li'),strong=document.createElement('strong');strong.textContent=h.state||'event';li.appendChild(strong);li.append(document.createTextNode(h.detail||''));history.appendChild(li)});$('live').className='live online';$('updated').textContent='LIVE · '+new Date(payload.server_time_unix_ms).toLocaleTimeString()}
async function refresh(){try{const response=await fetch('/api/snapshot',{cache:'no-store'});if(!response.ok)throw new Error('HTTP '+response.status);render(await response.json())}catch(error){$('live').className='live';$('updated').textContent='HUD DISCONNECTED'}setTimeout(refresh,200)}refresh();
</script>
</body>
</html>"""
