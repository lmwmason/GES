#include "flight_plan.h"
#include "math3d.h"

/* Scripted mission: takeoff, hover, then accelerate to and *hold* a
 * cruise speed for the rest of the flight. There is no GPS/optical-flow
 * in this airframe's sensor set, so speed is regulated open-loop by pitch
 * angle rather than a velocity PID -- drag grows with v^2, so a constant
 * pitch trim settles at a steady terminal speed instead of accelerating
 * forever (equilibrium: thrust*sin(pitch) == 0.5*rho*Cd*A*v^2).
 *
 * A single constant pitch for the whole cruise (this file's original
 * approach) turned out to badly conflate the two things a real
 * open-loop-pitch cruise needs: fast convergence wants a *large* pitch
 * (strong accelerating force), but a small, sane cruise speed wants a
 * *small* pitch (weak accelerating force to match the modest steady-state
 * drag). A single CRUISE_PITCH_RAD = 0.018 rad happened to converge inside
 * a ~60s episode, but only by overshooting to an ~6.9 m/s terminal speed
 * (verified against DP_BODY_DRAG_COEFF/DP_BODY_REF_AREA_M2) -- nowhere
 * near the "~3-4 m/s cruise" this file used to claim, and it was still
 * visibly accelerating at t=60s (v(t) = v_term*tanh(t/tau), tau~39s here).
 * The continuously-rising speed meant the vehicle swept the (fixed-length)
 * terrain features faster and faster as an episode went on, so the
 * altitude loop saw a rising disturbance frequency and its tracking
 * error grew across every episode regardless of ground-effect
 * feedforward -- not a real flight phenomenon, just this profile's own
 * never-converged transient (see docs/research_report.md Section 5.7).
 *
 * Fixed with two pitch values instead of one: a brief, larger ACCEL_PITCH_RAD
 * burst to reach CRUISE_SPEED_MPS quickly (open-loop timing -- the drag
 * physics are deterministic and terrain-independent, so this is
 * reproducible without any airspeed sensor), then a *fast* handoff (0.3s)
 * down to CRUISE_PITCH_RAD, the small trim pitch whose *own* equilibrium
 * speed is exactly CRUISE_SPEED_MPS. The handoff has to be fast: because
 * the trim pitch is so small, its restoring force is weak (time constant
 * ~80s), so any extra time spent above it before handoff overshoots by
 * roughly that much extra accumulated speed, and it would then take
 * many tens of seconds to bleed back off. Timings below were solved
 * numerically (integrating this same thrust-vs-drag equation in Python)
 * to land within ~6% of CRUISE_SPEED_MPS right at handoff and decay
 * gently from there -- see docs/research_report.md Section 5.7 for the
 * derivation; re-derive if DP_MASS_KG/DP_BODY_DRAG_COEFF/DP_BODY_REF_AREA_M2
 * ever change. */
/* Lowered from an earlier 0.35m: at that clearance the ground-effect
 * factor only reaches ~1.0-1.03 (see docs/research_report.md Section 5.6's
 * error-by-clearance figure), too small for any feedforward strategy's
 * correction to stand out above ordinary terrain-following noise in a
 * closed-loop comparison (Section 5.7's 8-seed result showed no
 * statistically significant difference between modes at 0.35m). 0.22m
 * keeps a collision margin above the 0.08m floor (rigid_body/main.c) while
 * pushing typical ground-effect factors into the ~1.05-1.15+ range where a
 * feedforward's compensation is large enough to plausibly separate from
 * noise. */
#define HOVER_AGL_M       0.22
#define CRUISE_SPEED_MPS  3.5
/* thrust*sin(pitch) = 0.5*rho*Cd*A*v^2 solved for pitch at v=CRUISE_SPEED_MPS,
 * using this airframe's DP_MASS_KG/DP_GRAVITY/DP_BODY_DRAG_COEFF/DP_BODY_REF_AREA_M2
 * (hand-solved rather than pulled from drone_params.h so this file has no
 * physics-engine dependency, matching the rest of the firmware). */
#define CRUISE_PITCH_RAD  0.00459
#define ACCEL_PITCH_RAD   0.15  /* ~8.6 deg: brief, larger pitch to reach cruise speed quickly */

#define T_TAKEOFF_END    3.0
#define T_HOVER_END      5.0
#define T_ACCEL_RAMP_IN  6.0  /* ramp 0 -> ACCEL_PITCH_RAD */
#define T_ACCEL_END      7.91 /* hold ACCEL_PITCH_RAD; numerically solved crossing of CRUISE_SPEED_MPS */
#define T_TRIM_RAMP_END  8.21 /* fast (0.3s) ramp ACCEL_PITCH_RAD -> CRUISE_PITCH_RAD */

flight_setpoint_t flight_plan_eval(double t) {
    flight_setpoint_t sp = {0.0, 0.0, 0.0, 0.0};
    sp.target_agl = HOVER_AGL_M;

    if (t < T_TAKEOFF_END) {
        sp.target_agl = HOVER_AGL_M * smoothstep01(t / T_TAKEOFF_END);
        sp.target_pitch = 0.0;
    } else if (t < T_HOVER_END) {
        sp.target_pitch = 0.0;
    } else if (t < T_ACCEL_RAMP_IN) {
        sp.target_pitch = ACCEL_PITCH_RAD * smoothstep01((t - T_HOVER_END) / (T_ACCEL_RAMP_IN - T_HOVER_END));
    } else if (t < T_ACCEL_END) {
        sp.target_pitch = ACCEL_PITCH_RAD;
    } else if (t < T_TRIM_RAMP_END) {
        double u = smoothstep01((t - T_ACCEL_END) / (T_TRIM_RAMP_END - T_ACCEL_END));
        sp.target_pitch = ACCEL_PITCH_RAD + (CRUISE_PITCH_RAD - ACCEL_PITCH_RAD) * u;
    } else {
        sp.target_pitch = CRUISE_PITCH_RAD; /* held for the rest of the flight */
    }

    sp.target_roll = 0.0;
    sp.target_yaw_rate = 0.0;
    return sp;
}
