#!/usr/bin/env python3
"""Log and calibrate thrust tests using the existing Rust Uno firmware."""

import argparse
import csv
import json
import math
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


GRAVITY = 9.80665
CALIBRATION_MS = 2000
MAX_PWM_US = 1500  # The Uno independently enforces its own cap.
TRIAL_COLUMNS = ("trial", "condition", "height_m", "target_rpm", "rpm", "thrust_N", "run_id")
SOUNDS = Path("/usr/share/sounds/freedesktop/stereo")
START_SOUND = "message-new-instant.oga"
STOP_SOUND = "complete.oga"


@dataclass(frozen=True)
class Sample:
    time_ms: int
    pwm_us: int
    raw_counts: int
    net_counts: int

    @classmethod
    def parse(cls, line):
        columns = line.split(",")
        if len(columns) == 4:
            try:
                values = tuple(map(int, columns))
            except ValueError:
                return None
            return cls(*values)
        return None


def fit_calibration(points):
    """Fit ADC counts as a linear function of applied force in newtons."""
    force = [point["mass_g"] / 1000 * GRAVITY for point in points]
    counts = [point["mean_counts"] for point in points]
    if len(set(force)) < 2 or 0.0 not in force:
        raise ValueError("Capture zero and at least one nonzero calibration mass.")
    slope, intercept = statistics.linear_regression(force, counts)
    if slope == 0:
        raise ValueError("No load-cell response to the calibration weights.")
    residual_n = [abs((y - intercept) / slope - x) for x, y in zip(force, counts)]
    return {"counts_per_n": slope, "zero_counts": intercept, "max_residual_n": max(residual_n)}


def to_newtons(counts, calibration):
    return (counts - calibration["zero_counts"]) / calibration["counts_per_n"]


def load_json(path, fallback):
    return json.loads(path.read_text()) if path.exists() else fallback


def save_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError("Expected a positive finite number.")
    return number


class Uno:
    def __init__(self, port):
        self.port = port

    def send(self, command):
        self.port.write((command + "\n").encode("ascii"))
        self.port.flush()

    def read(self):
        return self.port.readline().decode("ascii", errors="replace").strip()

    def response(self, prefix, seconds=2):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            line = self.read()
            if line.startswith(prefix):
                return line
            if line.startswith("# rejected"):
                raise RuntimeError(line)
        raise RuntimeError(f"No Uno response matching {prefix!r}.")

    def status(self):
        for attempt in range(3):
            self.send("STATUS")
            try:
                return self.response("# esc=", seconds=0.6)
            except RuntimeError:
                if attempt == 2:
                    raise

    def stop(self):
        state = "no STATUS response"
        for _ in range(2):
            self.send("STOP")
            time.sleep(1.2)
            try:
                state = self.status()
            except RuntimeError as exc:
                state = str(exc)
            if state.startswith("# esc=off pwm_us=0"):
                return
        raise RuntimeError(f"STOP could not be verified. Disconnect ESC power. Last state: {state}")

    def require_off(self, fresh=True):
        state = self.status()
        if not state.startswith("# esc=off pwm_us=0") or (
            fresh and not state.endswith("hx711_fresh=true")
        ):
            raise RuntimeError(f"Expected idle with valid HX711 input: {state}")


def capture_calibration(uno, directory, points, mass_g):
    if not math.isfinite(mass_g):
        raise ValueError("Calibration mass must be finite. Use a negative value for a reverse load.")
    uno.port.reset_input_buffer()
    uno.require_off()
    samples = []
    deadline = time.monotonic() + 3.5
    while time.monotonic() < deadline:
        sample = Sample.parse(uno.read())
        if sample is None:
            continue
        if sample.pwm_us != 0 or (samples and sample.time_ms <= samples[-1].time_ms):
            raise RuntimeError("PWM or clock changed during calibration.")
        samples.append(sample)
        if samples[-1].time_ms - samples[0].time_ms >= CALIBRATION_MS:
            break
    if len(samples) < 5 or samples[-1].time_ms - samples[0].time_ms < CALIBRATION_MS:
        raise RuntimeError("Load-cell samples did not cover the calibration window.")
    path = directory / f"cal_{len(points)+1:03}.csv"
    with path.open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time_ms", "raw_counts"))
        writer.writerows((sample.time_ms, sample.raw_counts) for sample in samples)
    counts = [sample.raw_counts for sample in samples]
    points.append({"mass_g": mass_g, "mean_counts": statistics.mean(counts),
                   "sd_counts": statistics.stdev(counts), "n": len(counts), "raw_file": path.name})
    save_json(directory / "calibration_points.json", points)
    print(f"{mass_g:g} g: {points[-1]['mean_counts']:.1f} counts "
          f"(SD {points[-1]['sd_counts']:.1f}, n={len(counts)})")


def trial_specs(args, duration, settle):
    block, condition, height_mm, target_rpm, pwm = args
    if condition not in ("near", "far"):
        raise ValueError("condition must be near or far.")
    pulse = int(pwm)
    if not 1001 <= pulse <= MAX_PWM_US:
        raise ValueError(f"PWM command must be 1001..{MAX_PWM_US} us.")
    return {"trial": block, "condition": condition, "height_m": positive(height_mm) / 1000,
            "target_rpm": positive(target_rpm), "pwm_us": pulse,
            "duration_ms": duration, "settle_ms": settle}


def recording_window(duration_ms, settle_ms):
    """Allow 2–3 s of camera startup and at least 1.25 s of steady footage."""
    start_ms = max(0, settle_ms - 2000)
    stop_ms = start_ms + 3000 + 1250
    if stop_ms + 500 > duration_ms:
        raise ValueError("Camera cues need at least 500 ms between the stop cue and the run end.")
    return start_ms, stop_ms


def recording_audio_command(sound):
    player = shutil.which("paplay")
    path = SOUNDS / sound
    if player is None or not path.is_file():
        raise RuntimeError("Camera cues require paplay and the freedesktop sound files.")
    return [player, str(path)]


def play_recording_cue(sound):
    subprocess.Popen(recording_audio_command(sound))


def run_trial(uno, directory, trials, specs, calibration):
    uno.port.reset_input_buffer()
    if uno.status() != "# esc=ready pwm_us=1000 hx711_fresh=true":
        raise RuntimeError("Use arm, wait for the ESC ready tone and check status before running.")
    run_id = f"run_{len(trials)+1:03}"
    path = directory / f"{run_id}.csv"
    record = {**specs, "run_id": run_id, "raw_file": path.name, "status": "started", "rpm": None,
              "created_utc": datetime.now(timezone.utc).isoformat(), "calibration": calibration}
    trials.append(record)
    save_json(directory / "trials.json", trials)
    first_ms = last_ms = None
    steady = []
    start_cue = stop_cue = None
    cue_label = ""
    try:
        with path.open("x", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("time_ms", "pwm_us", "raw_counts", "net_counts", "thrust_N", "steady"))
            pulse = specs["pwm_us"]
            duration_ms = specs["duration_ms"]
            uno.send(f"RUN {pulse} {duration_ms}")
            uno.response(f"# run: {pulse} us")
            deadline = time.monotonic() + duration_ms / 1000 + 1.0
            display = sys.stdout.isatty()
            if display:
                print("\x1b[2J\x1b[H", end="")
            print(f"{run_id}   {specs['trial']}   {specs['condition'].upper()}   "
                  f"{specs['height_m']*1000:g} mm   PWM {pulse} us", flush=True)
            if specs.get("propeller"):
                print(f"PROPELLER {specs['propeller']}", flush=True)
            if display:
                print("TIME [s]      THRUST [N]     PHASE        CAMERA", flush=True)
            observed_end = False
            while time.monotonic() < deadline:
                line = uno.read()
                if line.startswith(("# stopped", "# stop requested", "# thrust-bench-uno")):
                    raise RuntimeError(line)
                sample = Sample.parse(line)
                if sample is None:
                    continue
                if last_ms is not None and sample.time_ms <= last_ms:
                    raise RuntimeError("Uno timestamp reset or went backwards.")
                last_ms = sample.time_ms
                force = to_newtons(sample.raw_counts, calibration)
                if sample.pwm_us == pulse and first_ms is None:
                    first_ms = sample.time_ms
                elapsed = sample.time_ms - first_ms if first_ms is not None else 0
                keep = sample.pwm_us == pulse and elapsed >= specs["settle_ms"]
                writer.writerow((sample.time_ms, sample.pwm_us, sample.raw_counts,
                                 sample.net_counts, force, int(keep)))
                stream.flush()
                if sample.pwm_us != pulse and first_ms is not None:
                    observed_end = sample.pwm_us == 1000
                    break
                if sample.pwm_us == pulse and specs.get("camera_cues"):
                    start_ms, stop_ms = specs["camera_cues"]
                    if start_cue is None and elapsed >= start_ms:
                        play_recording_cue(START_SOUND)
                        start_cue = record["camera_start_elapsed_ms"] = elapsed
                        cue_label = "START RECORDING"
                    if start_cue is not None and stop_cue is None and elapsed >= stop_ms:
                        play_recording_cue(STOP_SOUND)
                        stop_cue = record["camera_stop_elapsed_ms"] = elapsed
                        cue_label = "STOP RECORDING"
                if keep:
                    steady.append(force)
                if display and sample.pwm_us == pulse:
                    print(f"\r{elapsed/1000:7.2f}       {force:+8.3f}      "
                          f"{'STEADY' if keep else 'SETTLING':8}     {cue_label:16}",
                          end="", flush=True)
            if display:
                print()
            # A sample rarely lands exactly on the start or end edge. Allow
            # two nominal HX711 intervals for sample/edge alignment.
            if not observed_end or first_ms is None or last_ms - first_ms < duration_ms - 250:
                raise RuntimeError("The observed PWM segment did not complete.")
            if len(steady) < 5:
                raise RuntimeError("Too few steady-window samples.")
            record.update(status="complete", thrust_N=statistics.mean(steady),
                          steady_sd_N=statistics.stdev(steady), steady_samples=len(steady))
    except BaseException as exc:
        record.update(status="incomplete", error=str(exc) or type(exc).__name__)
        raise
    finally:
        try:
            uno.stop()
        except Exception as exc:
            record.update(status="incomplete", stop_check=str(exc))
            raise RuntimeError("STOP could not be verified. Disconnect ESC power.") from exc
        finally:
            save_json(directory / "trials.json", trials)
            if start_cue is not None and stop_cue is None:
                play_recording_cue(STOP_SOUND)
    print(f"{run_id}: mean {record['thrust_N']:.3f} N. "
          f"Enter actual RPM with: rpm {run_id} <rpm>")


def export_measurements(directory, trials):
    ready = [row for row in trials if row["status"] == "complete" and row["rpm"] is not None]
    with (directory / "measurements.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=TRIAL_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ready)
    return len(ready)


def complete_run(trials, run_id):
    row = next((row for row in trials if row["run_id"] == run_id), None)
    if row is None or row["status"] != "complete":
        raise ValueError(f"{run_id}: use the ID of a complete run.")
    return row


def enter_rpm(directory, trials, run_id, rpm, source=None):
    row = complete_run(trials, run_id)
    row["rpm"] = positive(rpm)
    row["rpm_source"] = source or {"method": "manual"}
    save_json(directory / "trials.json", trials)
    return export_measurements(directory, trials)


def video_rpm(capture_fps, first_frame, last_frame, turns):
    fps = positive(capture_fps)
    frames = last_frame - first_frame
    if first_frame < 0 or frames <= 0 or turns <= 0:
        raise ValueError("Provide increasing frame indices and a positive count of complete turns.")
    if frames / turns < 4:
        raise ValueError("Fewer than four captured frames per revolution. Use a resolved recording.")
    return 60 * fps * turns / frames


def compare_pair(trials, far_id, near_id, tolerance):
    if not 0 < tolerance < 1:
        raise ValueError("RPM tolerance must be a fraction between 0 and 1.")
    far = complete_run(trials, far_id)
    near = complete_run(trials, near_id)
    if (far["condition"], near["condition"]) != ("far", "near"):
        raise ValueError("Provide the far run first and the near run second.")
    for field in ("trial", "target_rpm", "propeller", "duration_ms", "settle_ms", "calibration"):
        if far.get(field) != near.get(field):
            raise ValueError(f"The runs differ in {field}.")
    if not 0 < near["height_m"] < far["height_m"]:
        raise ValueError("The near run height must be below the positive far height.")
    if far["rpm"] is None or near["rpm"] is None:
        raise ValueError("Both runs need independently measured RPM.")
    far_rpm, near_rpm = positive(far["rpm"]), positive(near["rpm"])
    far_force, near_force = positive(far["thrust_N"]), positive(near["thrust_N"])
    rpm_difference = (near_rpm - far_rpm) / far_rpm
    result = {"far_run": far_id, "near_run": near_id, "propeller": far.get("propeller"),
              "far_height_m": far["height_m"], "near_height_m": near["height_m"],
              "far_rpm": far_rpm, "near_rpm": near_rpm, "relative_rpm_difference": rpm_difference,
              "rpm_tolerance": tolerance, "far_thrust_N": far_force, "near_thrust_N": near_force,
              "matched": abs(rpm_difference) <= tolerance}
    if result["matched"]:
        result["near_minus_far_N"] = near_force - far_force
        result["near_to_far_thrust_ratio"] = near_force / far_force
    return result


HELP = """cal 0                        Capture unloaded assembled-fixture readings
cal <signed-grams>           Positive for thrust direction, negative for reverse
fit                          Compute counts/N from calibration
arm                          Output minimum PWM only. Wait for the ESC ready tone.
run <block> <near|far> <height-mm> <target-rpm> <pwm-us>
rpm <run-id> <actual-rpm>    Enter independently measured RPM
status                       Read Uno state
stop                         Stop and verify that PWM is off
q                            Stop and exit (Ctrl+C also stops a run)"""


def console(args):
    import serial

    cues = None
    if args.camera_cues:
        cues = recording_window(args.duration_ms, args.settle_ms)
        for sound in (START_SOUND, STOP_SOUND):
            recording_audio_command(sound)

    points = load_json(args.out / "calibration_points.json", [])
    trials = load_json(args.out / "trials.json", [])
    calibration = load_json(args.out / "calibration.json", None)
    with serial.Serial(args.port, 115200, timeout=0.17, write_timeout=1, exclusive=True) as port:
        time.sleep(2.5)
        uno = Uno(port)
        port.reset_input_buffer()
        uno.require_off()
        print(f"Session: {args.out.resolve()}\n{HELP}")
        if cues:
            print(f"Camera sounds at {cues[0]/1000:g} s (start) and {cues[1]/1000:g} s (stop).")
        while True:
            try:
                words = input("thrust> ").strip().split()
                if not words:
                    continue
                cmd, *rest = words
                match cmd.lower(), len(rest):
                    case ("q", 0):
                        break
                    case ("status", 0):
                        port.reset_input_buffer()
                        print(uno.status())
                    case ("cal", 1):
                        capture_calibration(uno, args.out, points, float(rest[0]))
                        calibration = None
                        save_json(args.out / "calibration.json", None)
                    case ("fit", 0):
                        calibration = fit_calibration(points)
                        save_json(args.out / "calibration.json", calibration)
                        print(json.dumps(calibration, indent=2))
                    case ("arm", 0) if calibration is not None:
                        port.reset_input_buffer()
                        uno.require_off()
                        uno.send("ARM")
                        print(uno.response("# arming:"))
                        print("Verify the ESC ready tone and use status. Type stop if the tone is absent.")
                    case ("stop", 0):
                        port.reset_input_buffer()
                        uno.stop()
                        print("PWM off.")
                    case ("run", 5) if calibration is not None:
                        specs = trial_specs(rest, args.duration_ms, args.settle_ms)
                        if args.propeller:
                            specs["propeller"] = args.propeller
                        if cues:
                            specs["camera_cues"] = cues
                        run_trial(uno, args.out, trials, specs, calibration)
                    case ("rpm", 2):
                        n = enter_rpm(args.out, trials, *rest)
                        print(f"{n} rows in measurements.csv")
                    case _:
                        print("Fit a calibration before RUN.\n" + HELP)
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                break
            except (ValueError, RuntimeError) as exc:
                print(f"Error: {exc}", file=sys.stderr)
                if "Disconnect ESC power" in str(exc):
                    raise
        uno.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("start")
    start.add_argument("--port", help="Serial port, defaults to the saved port or /dev/ttyACM0.")
    output = start.add_mutually_exclusive_group()
    output.add_argument("--out", type=Path)
    output.add_argument("--resume", type=Path, help="Resume an existing session with the same fixture.")
    start.add_argument("--duration-ms", type=int, default=5000)
    start.add_argument("--settle-ms", type=int, default=2000)
    start.add_argument("--propeller", help="Propeller identifier to display and store with each run.")
    start.add_argument("--camera-cues", action="store_true", help="Play recording start/stop sounds.")
    commands.add_parser("test-cues", help="Play the two camera sounds without opening the Uno port.")
    annotate = commands.add_parser("rpm", help="Record measured RPM after closing the console.")
    annotate.add_argument("session", type=Path)
    annotate.add_argument("run_id")
    annotate.add_argument("actual_rpm")
    video = commands.add_parser("video-rpm", help="Record RPM from a steady video segment.")
    video.add_argument("session", type=Path)
    video.add_argument("run_id")
    video.add_argument("--video", required=True, help="Source video filename")
    video.add_argument("--capture-fps", type=float, required=True)
    video.add_argument("--first-frame", type=int, required=True)
    video.add_argument("--last-frame", type=int, required=True)
    video.add_argument("--turns", type=int, required=True)
    pair = commands.add_parser("pair", help="Compare two runs at matched measured RPM.")
    pair.add_argument("session", type=Path)
    pair.add_argument("far_id")
    pair.add_argument("near_id")
    pair.add_argument("--rpm-tolerance", type=float, required=True)
    args = parser.parse_args()
    if args.command == "test-cues":
        for label, sound in (("START RECORDING", START_SOUND), ("STOP RECORDING", STOP_SOUND)):
            print(label, flush=True)
            subprocess.run(recording_audio_command(sound), check=True)
            time.sleep(0.35)
        return
    if args.command != "start":
        trials = load_json(args.session / "trials.json", [])
        try:
            if args.command == "pair":
                print(json.dumps(compare_pair(trials, args.far_id, args.near_id, args.rpm_tolerance),
                                 indent=2, allow_nan=False))
            elif args.command == "video-rpm":
                rpm = video_rpm(args.capture_fps, args.first_frame, args.last_frame, args.turns)
                source = {"method": "video", "video": args.video, "capture_fps": args.capture_fps,
                          "first_frame": args.first_frame, "last_frame": args.last_frame,
                          "turns": args.turns}
                n = enter_rpm(args.session, trials, args.run_id, rpm, source)
                print(f"{args.run_id}: {rpm:.2f} rpm, {n} rows in measurements.csv")
            else:
                n = enter_rpm(args.session, trials, args.run_id, args.actual_rpm)
                print(f"{n} rows in measurements.csv")
        except (ValueError, KeyError) as exc:
            parser.error(str(exc))
        return
    if args.resume is not None:
        session = load_json(args.resume / "session.json", None)
        if session is None:
            parser.error("resume requires an existing session directory")
        args.out = args.resume
        args.port = args.port or session["port"]
        args.duration_ms = session["duration_ms"]
        args.settle_ms = session["settle_ms"]
        args.propeller = args.propeller or session.get("propeller")
        console(args)
        return
    if not 1000 <= args.duration_ms <= 10000 or not 0 <= args.settle_ms <= args.duration_ms - 700:
        parser.error("duration 1000–10000 ms, with at least 700 ms after settling")
    if args.out is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        args.out = Path("runs") / stamp
    args.port = args.port or "/dev/ttyACM0"
    args.out.mkdir(parents=True, exist_ok=False)
    save_json(args.out / "session.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
              "port": args.port, "duration_ms": args.duration_ms, "settle_ms": args.settle_ms,
              "propeller": args.propeller})
    console(args)


if __name__ == "__main__":
    main()
