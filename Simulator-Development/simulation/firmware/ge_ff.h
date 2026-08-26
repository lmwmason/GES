#ifndef GE_FF_H
#define GE_FF_H

/* Ground-effect feedforward compensation, computed entirely from what the
 * firmware can actually sense (a single downward rangefinder -- no
 * knowledge of terrain shape). This is intentionally a coarser estimate
 * than the physics engine's ground truth (physics/aero_ground_effect.c),
 * which also accounts for local slope/curvature: the whole point of the
 * future AI model is to close that gap using cues the analytical formula
 * can't see (motor current anomaly + IMU pattern). */

typedef enum {
    GE_FF_SRC_ANALYTICAL = 0,
    GE_FF_SRC_AI = 1,
    GE_FF_SRC_DISABLED = 2,
    GE_FF_SRC_ORACLE = 3
} ge_ff_source_t;

/* Which feedforward compensation fc_step() applies. Selectable at runtime
 * via the GE_FF_MODE environment variable (see ge_ff_mode_from_env())
 * purely so one compiled firmware can be A/B/C-tested against itself --
 * see ml/analysis/ff_mode_comparison.py, which runs the simulator three
 * times under the three modes and compares the resulting IMU traces. */
typedef enum {
    GE_FF_MODE_NONE = 0,       /* no feedforward: PID reacts to ground effect purely via feedback */
    GE_FF_MODE_ANALYTICAL = 1, /* always use the flat-ground Cheeseman-Bennett estimate */
    GE_FF_MODE_ML = 2,         /* hybrid: ML with low-clearance analytical cutover */
    GE_FF_MODE_ML_PURE = 3,    /* ML with the standard gate and no cutover */
    GE_FF_MODE_ML_NARROW = 4,  /* ML with a narrow gate and no cutover */
    GE_FF_MODE_ORACLE = 5,     /* diagnostic only: simulator ground truth */
    GE_FF_MODE_ML_DIRECT = 6,  /* direct/ungated MLP prediction */
    GE_FF_MODE_ML_ASYM_CLAMP = 7 /* hybrid plus upper cap vs analytical */
} ge_ff_mode_t;

#define GE_FF_TRUST_BAND 0.20 /* +/-20% */
#define GE_FF_NARROW_TRUST_BAND 0.05 /* +/-5%, corrected-mapping ablation */

/* Below this clearance the hybrid skips ML and uses the analytical estimate.
 * In the corrected-mapping ablation this outperformed both pure ML and a
 * narrowed trust band, whose reject path sets factor=1.0 (no compensation). */
#define GE_FF_LOW_CLEARANCE_M       0.15

/* Reads GE_FF_MODE ("none" | "analytical" | "ml", case-insensitive) from
 * the environment; defaults to GE_FF_MODE_ANALYTICAL if unset or
 * unrecognized (see the .c file for why ML isn't the default despite
 * scoring far better offline -- docs/research_report.md Section 5.9). */
ge_ff_mode_t ge_ff_mode_from_env(void);

/* Flat, level-ground Cheeseman-Bennett estimate from rangefinder altitude. */
double ge_ff_analytical_factor(double rangefinder_alt_m);

/* Trained MLP (see ml/models_torch.py:MLPNet, ml/export_mlp_to_c.py,
 * firmware/ge_ff_ml_weights.h) forward pass. `features` must be packed in
 * exactly the order documented at the top of ge_ff_ml_weights.h: gyro
 * xyz, rangefinder_alt, motor_rpm x4, motor_current x4, thrust_cmd_norm
 * (the *previous* tick's commanded throttle -- this tick's isn't known
 * yet at the point this is called, see fc_step()). Returns NAN if
 * rangefinder_alt is NAN (out of range), same as the analytical model's
 * fallback, which ge_ff_select() below will treat as "don't trust this". */
double ge_ff_ml_predict(const double features[13]);

/* Selects analytical vs AI vs disabled: if |ai_pred - analytical| stays
 * within band_frac of analytical, trust the AI value; otherwise fall back
 * to disabling feedforward entirely for this tick (never blend/clamp --
 * an FF term the controller can't trust is worse than none). */
ge_ff_source_t ge_ff_select(double analytical, double ai_pred, double band_frac, double *out_factor);

/* Correct actuator-domain mapping for a thrust-domain GE multiplier. */
double ge_ff_compensate_throttle(double throttle_raw, double factor);
double ge_ff_asym_clamp_delta_from_env(void);

#endif
