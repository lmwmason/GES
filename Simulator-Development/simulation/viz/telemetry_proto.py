"""
Binary layout matching sim/telemetry.h's telemetry_packet_t exactly
(packed, no padding). Keep in sync by hand if that struct changes.
"""
import struct
from dataclasses import dataclass

# t(d) pos(3f) quat(4f) vel(3f) ang_vel(3f) motor_rpm(4f) motor_thrust(4f)
# ge_factor_truth(4f) rangefinder_alt(f) ge_ff_analytical(f) ge_ff_ai(f)
# ge_ff_source(i) thrust_cmd_norm(f)
FORMAT = "<d3f4f3f3f4f4f4ffffif"
PACKET_SIZE = struct.calcsize(FORMAT)

GE_SRC_ANALYTICAL, GE_SRC_AI, GE_SRC_DISABLED = 0, 1, 2
GE_SRC_NAMES = {0: "ANALYTICAL", 1: "AI", 2: "DISABLED"}


@dataclass
class Telemetry:
    t: float
    pos: tuple
    quat: tuple
    vel: tuple
    ang_vel: tuple
    motor_rpm: tuple
    motor_thrust: tuple
    ge_factor_truth: tuple
    rangefinder_alt: float
    ge_ff_analytical: float
    ge_ff_ai: float
    ge_ff_source: int
    thrust_cmd_norm: float


def unpack(data: bytes) -> Telemetry:
    vals = struct.unpack(FORMAT, data[:PACKET_SIZE])
    i = 0
    def take(n):
        nonlocal i
        v = vals[i:i + n]
        i += n
        return v

    t = take(1)[0]
    pos = take(3)
    quat = take(4)
    vel = take(3)
    ang_vel = take(3)
    motor_rpm = take(4)
    motor_thrust = take(4)
    ge_factor_truth = take(4)
    rangefinder_alt = take(1)[0]
    ge_ff_analytical = take(1)[0]
    ge_ff_ai = take(1)[0]
    ge_ff_source = take(1)[0]
    thrust_cmd_norm = take(1)[0]

    return Telemetry(t, pos, quat, vel, ang_vel, motor_rpm, motor_thrust,
                      ge_factor_truth, rangefinder_alt, ge_ff_analytical,
                      ge_ff_ai, ge_ff_source, thrust_cmd_norm)
