#include "fc_api.h"
#include "sensor_types.h"
#include "drone_params.h"
#include "math3d.h"
#include "attitude_estimator.h"
#include "pid.h"
#include "mixer.h"
#include "flight_plan.h"
#include "ge_ff.h"
#include "ge_ff_ml_weights.h"
#include <math.h>
#include <stdio.h>

/* This file is the whole "external firmware": everything it knows comes
 * through sensor_packet_t, everything it produces is actuator_packet_t.
 * It is built as libfc.so and dlopen()'d by the simulator (see sim/main.c)
 * rather than linked in, so it can be swapped like a real flashed image. */

typedef struct {
    attitude_estimator_t ae;
    pid_t pid_roll_att, pid_pitch_att;
    pid_t pid_roll_rate, pid_pitch_rate, pid_yaw_rate;
    pid_t pid_alt;
    double hover_cmd;
    ge_ff_mode_t ge_ff_mode;
    fc_debug_t dbg;
} fc_state_t;

static fc_state_t g;

void fc_init(void) {
    ae_init(&g.ae);

    /* attitude (outer, P-only): rad error -> rad/s rate command */
    pid_init(&g.pid_roll_att, 4.0, 0.0, 0.0, 0.0, 3.0, 0.0);
    pid_init(&g.pid_pitch_att, 4.0, 0.0, 0.0, 0.0, 3.0, 0.0);

    /* rate (inner): rad/s error -> normalized mixer trim. Gyro is fairly
     * clean, but still filter the D term lightly. */
    pid_init(&g.pid_roll_rate, 0.060, 0.020, 0.0020, 0.30, 0.40, 0.010);
    pid_init(&g.pid_pitch_rate, 0.060, 0.020, 0.0020, 0.30, 0.40, 0.010);
    pid_init(&g.pid_yaw_rate, 0.080, 0.010, 0.0, 0.30, 0.30, 0.0);

    /* altitude hold: meters error -> normalized throttle trim around hover.
     * Rangefinder is noisy relative to the 1kHz loop rate, so the D term
     * needs a much heavier filter than the rate loops. */
    pid_init(&g.pid_alt, 0.60, 0.15, 0.25, 0.20, 0.40, 0.15);

    double hover_thrust_per_motor = (DP_MASS_KG * DP_GRAVITY) / 4.0;
    double hover_rpm = sqrt(hover_thrust_per_motor / DP_MOTOR_KT);
    g.hover_cmd = hover_rpm / DP_MOTOR_RPM_MAX;

    g.ge_ff_mode = ge_ff_mode_from_env();
    g.dbg = (fc_debug_t){0};
}

void fc_step(const sensor_packet_t *in, actuator_packet_t *out) {
    ae_update(&g.ae, in->gyro, in->accel, in->dt);

    flight_setpoint_t sp = flight_plan_eval(in->t);

    double roll_rate_cmd = pid_update(&g.pid_roll_att, sp.target_roll - g.ae.roll, in->dt);
    double pitch_rate_cmd = pid_update(&g.pid_pitch_att, sp.target_pitch - g.ae.pitch, in->dt);
    double yaw_rate_cmd = sp.target_yaw_rate;

    double roll_cmd = pid_update(&g.pid_roll_rate, roll_rate_cmd - in->gyro[0], in->dt);
    double pitch_cmd = pid_update(&g.pid_pitch_rate, pitch_rate_cmd - in->gyro[1], in->dt);
    double yaw_cmd = pid_update(&g.pid_yaw_rate, yaw_rate_cmd - in->gyro[2], in->dt);

    double agl = in->rangefinder_alt;
    if (isnan(agl)) agl = in->baro_alt;
    double alt_err = sp.target_agl - agl;
    double throttle_trim = pid_update(&g.pid_alt, alt_err, in->dt);
    double throttle_raw = g.hover_cmd + throttle_trim;

    double analytical = ge_ff_analytical_factor(in->rangefinder_alt);
    double ge_factor = 1.0;
    ge_ff_source_t src;

    switch (g.ge_ff_mode) {
    case GE_FF_MODE_NONE:
        ge_factor = 1.0;
        src = GE_FF_SRC_DISABLED;
        g.dbg.ge_ff_ai = NAN;
        break;
    case GE_FF_MODE_ANALYTICAL:
        ge_factor = analytical;
        src = GE_FF_SRC_ANALYTICAL;
        g.dbg.ge_ff_ai = NAN;
        break;
    case GE_FF_MODE_ML:
    default: {
        /* Feature order must match ge_ff_ml_weights.h exactly. thrust_cmd_norm
         * is the *previous* tick's g.dbg value (this tick's throttle_cmd is
         * what we're computing right now, so it can't be its own input --
         * the two are close enough tick-to-tick at 60Hz that the one-tick
         * lag costs negligible accuracy, see docs/research_report.md). */
        double features[GE_FF_ML_N_FEATURES] = {
            in->gyro[0], in->gyro[1], in->gyro[2], in->rangefinder_alt,
            in->motor_rpm_meas[0], in->motor_rpm_meas[1], in->motor_rpm_meas[2], in->motor_rpm_meas[3],
            in->motor_current[0], in->motor_current[1], in->motor_current[2], in->motor_current[3],
            g.dbg.thrust_cmd_norm,
        };
        double ai_pred = ge_ff_ml_predict(features);
        if (in->rangefinder_alt < GE_FF_LOW_CLEARANCE_M) {
            /* Don't just tighten the trust band here: ge_ff_select()'s
             * reject path sets factor=1.0 (no compensation at all), which
             * is worse than a crude-but-safe analytical estimate right
             * where compensation matters most (confirmed mechanism,
             * docs/research_report.md Section 5.11) -- so below this
             * clearance, skip the AI path entirely and use analytical. */
            ge_factor = analytical;
            src = GE_FF_SRC_ANALYTICAL;
        } else {
            src = ge_ff_select(analytical, ai_pred, GE_FF_TRUST_BAND, &ge_factor);
        }
        g.dbg.ge_ff_ai = ai_pred;
        break;
    }
    }

    double throttle_cmd = throttle_raw;
    if (src != GE_FF_SRC_DISABLED) {
        /* Pre-divide by the expected thrust augmentation so the altitude
         * loop doesn't have to react after the fact. */
        throttle_cmd = throttle_raw / ge_factor;
    }
    throttle_cmd = clampd(throttle_cmd, 0.0, 1.0);

    mixer_compute(throttle_cmd, roll_cmd, pitch_cmd, yaw_cmd, out->motor_cmd);

    g.dbg.ge_ff_analytical = analytical;
    g.dbg.ge_ff_source = (int)src;
    g.dbg.thrust_cmd_norm = throttle_cmd;

#ifdef FC_DEBUG_PRINT
    static int dbgcnt = 0;
    if (++dbgcnt % 200 == 0) {
        fprintf(stderr, "[fc] t=%.2f roll=%.4f pitch=%.4f yaw=%.4f gyro=(%.4f,%.4f,%.4f) rcmd=(%.3f,%.3f,%.3f) mcmd=(%.3f,%.3f,%.3f,%.3f)\n",
                in->t, g.ae.roll, g.ae.pitch, g.ae.yaw,
                in->gyro[0], in->gyro[1], in->gyro[2],
                roll_cmd, pitch_cmd, yaw_cmd,
                out->motor_cmd[0], out->motor_cmd[1], out->motor_cmd[2], out->motor_cmd[3]);
    }
#endif
}

void fc_get_debug(fc_debug_t *out) {
    *out = g.dbg;
}
