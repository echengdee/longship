#!/usr/bin/env python3
"""Serve remote-friendly live views of camera and exact policy debug inputs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time
from typing import Any
from urllib.parse import urlparse


@dataclass(slots=True)
class Frame:
    jpeg: bytes
    metadata: dict[str, Any]
    sequence: int


class FrameStore:
    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.frames: dict[str, Frame] = {}
        self.sequence = 0

    def update(self, channel: str, metadata: dict[str, Any], jpeg: bytes) -> None:
        with self.condition:
            self.sequence += 1
            self.frames[channel] = Frame(jpeg, metadata, self.sequence)
            self.condition.notify_all()

    def wait(self, channel: str, sequence: int, timeout: float = 2.0) -> Frame | None:
        deadline = time.monotonic() + timeout
        with self.condition:
            while time.monotonic() < deadline:
                frame = self.frames.get(channel)
                if frame is not None and frame.sequence > sequence:
                    return frame
                self.condition.wait(deadline - time.monotonic())
        return None

    def status(self) -> dict[str, Any]:
        with self.condition:
            return {
                channel: {**frame.metadata, "sequence": frame.sequence}
                for channel, frame in self.frames.items()
            }


HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Longship RL Vision</title><style>
body{margin:0;background:#101318;color:#e8edf2;font:15px system-ui,sans-serif}header{padding:18px 24px;border-bottom:1px solid #29313b}
h1{font-size:20px;margin:0 0 5px}.sub{color:#91a0af}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:18px;padding:18px}
.card{background:#171c22;border:1px solid #29313b;border-radius:10px;overflow:hidden}.title{padding:12px 14px;font-weight:650}.view{background:#050607;min-height:260px;display:flex;align-items:center;justify-content:center}
img{width:100%;height:auto;image-rendering:auto}.meta{padding:10px 14px;color:#9facb9;font:13px ui-monospace,monospace;white-space:pre-wrap}
.ok{color:#63d392}.warn{color:#f2c66d}</style></head><body><header><h1>Longship RL Vision</h1><div class="sub">D435i 原始深度与 Hiking 实际模型输入</div></header>
<main class="grid"><section class="card"><div class="title">相机深度 · camera_depth</div><div class="view"><img src="/stream/camera_depth"></div><div class="meta" id="camera_depth">等待数据…</div></section>
<section class="card"><div class="title">模型输入 · model_depth</div><div class="view"><img src="/stream/model_depth"></div><div class="meta" id="model_depth">等待策略推理…</div></section></main>
<script>async function poll(){try{let s=await (await fetch('/api/status')).json(),now=Date.now()/1000;for(let id of ['camera_depth','model_depth']){let e=document.getElementById(id),v=s[id];if(!v){e.textContent='等待数据…';e.className='meta warn';continue}let age=now-v.timestamp;e.textContent=`shape=${v.shape.join('×')}  range=${Number(v.min).toFixed(3)}..${Number(v.max).toFixed(3)}  age=${age.toFixed(2)}s`;e.className='meta '+(age<1?'ok':'warn')}}catch(e){}}setInterval(poll,500);poll()</script></body></html>"""


def _receive(socket: Any, store: FrameStore) -> None:
    while True:
        channel, metadata, jpeg = socket.recv_multipart()
        try:
            decoded = json.loads(metadata)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        store.update(channel.decode(errors="replace"), decoded, jpeg)


def handler_for(store: FrameStore) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/":
                self._reply(200, "text/html; charset=utf-8", HTML.encode())
            elif path == "/api/status":
                self._reply(200, "application/json", json.dumps(store.status()).encode())
            elif path.startswith("/stream/"):
                self._stream(path.removeprefix("/stream/"))
            else:
                self._reply(404, "text/plain", b"not found")

        def _reply(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _stream(self, channel: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            sequence = 0
            try:
                while True:
                    frame = store.wait(channel, sequence)
                    if frame is None:
                        continue
                    sequence = frame.sequence
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(frame.jpeg)).encode()
                        + b"\r\n\r\n"
                        + frame.jpeg
                        + b"\r\n"
                    )
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind-host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--frame-endpoint", default="tcp://127.0.0.1:5570")
    args = parser.parse_args()
    store = FrameStore()
    import zmq

    socket = zmq.Context.instance().socket(zmq.PULL)
    socket.setsockopt(zmq.RCVHWM, 4)
    socket.bind(args.frame_endpoint)
    threading.Thread(target=_receive, args=(socket, store), daemon=True).start()
    server = ThreadingHTTPServer((args.bind_host, args.port), handler_for(store))
    print(
        f"WEB MONITOR READY: url=http://{args.bind_host}:{args.port} "
        f"frames={args.frame_endpoint}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
