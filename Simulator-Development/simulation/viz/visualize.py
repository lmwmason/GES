#!/usr/bin/env python3
"""
Real-time 3D viewer for the drone/ground-effect simulator.

Listens on a UDP socket for telemetry_packet_t frames sent by the C
simulator (sim/main.c), and renders the vehicle flying over the terrain
height field in a pyqtgraph OpenGL view, alongside a live readout of the
firmware's ground-effect feedforward state (analytical vs AI vs disabled).

Usage:
    python3 viz/visualize.py [--port 5555] [--host 0.0.0.0]
"""
import argparse
import socket
import sys

import numpy as np
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from terrain_ref import terrain_height_grid, generate_features
from telemetry_proto import unpack, PACKET_SIZE, GE_SRC_NAMES

# Body-frame rotor offsets, must match common/drone_params.h DP_MOTOR_LAYOUT.
ARM = 0.18
S = 0.70710678
ROTOR_OFFSETS = np.array([
    [ ARM * S, -ARM * S, 0.0],  # 0: front-right, CW
    [-ARM * S,  ARM * S, 0.0],  # 1: rear-left,   CW
    [ ARM * S,  ARM * S, 0.0],  # 2: front-left,  CCW
    [-ARM * S, -ARM * S, 0.0],  # 3: rear-right,  CCW
])

TRAIL_MAX_LEN = 2000


def quat_to_rotmat(w, x, y, z):
    n = w*w + x*x + y*y + z*z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    wx, wy, wz = s*w*x, s*w*y, s*w*z
    xx, xy, xz = s*x*x, s*x*y, s*x*z
    yy, yz, zz = s*y*y, s*y*z, s*z*z
    return np.array([
        [1 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1 - (xx + yy)],
    ])


def ge_color(factor):
    """Green at factor=1 (no ground effect), sweeping to red as it rises."""
    t = np.clip((factor - 1.0) / 0.35, 0.0, 1.0)
    return (t, 1.0 - t, 0.15, 1.0)


class DroneViewer(QtWidgets.QMainWindow):
    def __init__(self, host, port, seed):
        super().__init__()
        self.seed = seed
        self.setWindowTitle("Drone Ground-Effect Simulator")
        self.resize(1280, 800)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)
        self.setCentralWidget(central)

        self.gl_view = gl.GLViewWidget()
        self.gl_view.setCameraPosition(distance=25, elevation=25, azimuth=-60)
        layout.addWidget(self.gl_view, stretch=4)

        self.hud = QtWidgets.QLabel("waiting for telemetry...")
        self.hud.setStyleSheet(
            "font-family: monospace; font-size: 13px; padding: 10px;"
        )
        self.hud.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        panel = QtWidgets.QScrollArea()
        panel.setWidget(self.hud)
        panel.setWidgetResizable(True)
        panel.setMinimumWidth(320)
        panel.setMaximumWidth(380)
        layout.addWidget(panel, stretch=1)

        self._build_terrain()
        self._build_drone()

        self.trail_pts = np.zeros((TRAIL_MAX_LEN, 3))
        self.trail_len = 0
        self.trail = gl.GLLinePlotItem(pos=self.trail_pts[:1], color=(0.2, 0.6, 1.0, 0.8), width=2, antialias=True)
        self.gl_view.addItem(self.trail)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.sock.setblocking(False)

        self.latest = None
        self.packet_count = 0

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(16)  # ~60 Hz

    def _build_terrain(self):
        features = generate_features(self.seed)
        x_max = max((f[1] + f[2] for f in features), default=60.0) + 5.0
        y_max = max((abs(f[4]) + 2.5 * f[5] for f in features), default=6.0)
        y_max = max(y_max, 6.0)

        # Tracks can now run for kilometers (hundreds-to-1500+ features),
        # so cap the static overview mesh to a fixed point budget rather
        # than a fixed spacing -- otherwise a huge track would make this
        # one-time mesh build take minutes. This is a coarse full-track
        # overview, not a per-frame render.
        n_x = min(int((x_max + 5) / 0.5) + 1, 600)
        n_y = min(int(2 * y_max / 0.5) + 1, 41)
        xs = np.linspace(-5, x_max, n_x)
        ys = np.linspace(-y_max, y_max, n_y)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        Z = terrain_height_grid(X, Y, self.seed)

        surf = gl.GLSurfacePlotItem(x=xs, y=ys, z=Z, shader="shaded",
                                     color=(0.45, 0.48, 0.55, 1.0))
        surf.setGLOptions("opaque")
        self.gl_view.addItem(surf)

        grid = gl.GLGridItem()
        grid.setSize(x=x_max + 5, y=2 * y_max)
        grid.setSpacing(x=2, y=2)
        grid.translate(x_max / 2, 0, -0.5)
        self.gl_view.addItem(grid)

    def _build_drone(self):
        # Cross-frame lines (center -> each rotor) + a rotor scatter marker.
        self.frame_lines = []
        for _ in range(4):
            item = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=(1, 1, 1, 1), width=3)
            self.gl_view.addItem(item)
            self.frame_lines.append(item)

        self.rotor_scatter = gl.GLScatterPlotItem(pos=np.zeros((4, 3)), size=14, color=(0, 1, 0, 1))
        self.gl_view.addItem(self.rotor_scatter)

    def tick(self):
        latest_data = None
        try:
            while True:
                data, _ = self.sock.recvfrom(4096)
                if len(data) >= PACKET_SIZE:
                    latest_data = data
                    self.packet_count += 1
        except BlockingIOError:
            pass

        if latest_data is None:
            return

        tel = unpack(latest_data)
        self.latest = tel
        self.update_scene(tel)
        self.update_hud(tel)

    def update_scene(self, tel):
        pos = np.array(tel.pos, dtype=float)
        R = quat_to_rotmat(*tel.quat)
        rotor_world = pos + ROTOR_OFFSETS @ R.T

        for i in range(4):
            self.frame_lines[i].setData(pos=np.vstack([pos, rotor_world[i]]))

        colors = np.array([ge_color(f) for f in tel.ge_factor_truth])
        self.rotor_scatter.setData(pos=rotor_world, color=colors, size=14)

        if self.trail_len < TRAIL_MAX_LEN:
            self.trail_pts[self.trail_len] = pos
            self.trail_len += 1
        else:
            self.trail_pts[:-1] = self.trail_pts[1:]
            self.trail_pts[-1] = pos
        self.trail.setData(pos=self.trail_pts[:self.trail_len])

        cam_center = pg.Vector(pos[0], pos[1], pos[2])
        self.gl_view.opts["center"] = cam_center
        self.gl_view.update()

    def update_hud(self, tel):
        src_name = GE_SRC_NAMES.get(tel.ge_ff_source, "?")
        lines = [
            f"t             = {tel.t:7.2f} s",
            f"packets       = {self.packet_count}",
            "",
            f"pos (x,y,z)   = {tel.pos[0]:6.2f}, {tel.pos[1]:6.2f}, {tel.pos[2]:6.2f} m",
            f"vel (x,y,z)   = {tel.vel[0]:6.2f}, {tel.vel[1]:6.2f}, {tel.vel[2]:6.2f} m/s",
            f"rangefinder   = {tel.rangefinder_alt:6.3f} m",
            "",
            "-- ground effect --",
            f"truth factor  = " + ", ".join(f"{f:.3f}" for f in tel.ge_factor_truth),
            f"analytical FF = {tel.ge_ff_analytical:6.3f}",
            f"AI-hook FF    = {tel.ge_ff_ai:6.3f}",
            f"FF source     = {src_name}",
            f"thrust cmd    = {tel.thrust_cmd_norm:5.3f}",
            "",
            "-- motors (rpm) --",
        ]
        for i in range(4):
            lines.append(f"  m{i}: {tel.motor_rpm[i]:7.0f} rpm  {tel.motor_thrust[i]:5.2f} N")

        self.hud.setText("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--seed", type=int, default=1,
                     help="terrain seed, must match the simulator's --seed")
    args = ap.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOption("background", (18, 20, 26))
    win = DroneViewer(args.host, args.port, args.seed)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
