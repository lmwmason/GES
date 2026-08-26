#include "ge_ff.h"
#include "ge_ff_ml_weights.h"
#include "drone_params.h"
#include "math3d.h"
#include <math.h>
#include <stdlib.h>
#include <strings.h>

/* Flat-ground Cheeseman-Bennett thrust multiplier. The controller converts
 * this thrust-domain factor to an actuator-domain correction with 1/sqrt(f)
 * because motor thrust is quadratic in the throttle-proportional rpm. */
double ge_ff_analytical_factor(double rangefinder_alt_m) {
    if (isnan(rangefinder_alt_m) || rangefinder_alt_m <= 0.0) return 1.0;
    double z = rangefinder_alt_m;
    double z_min = DP_ROTOR_RADIUS_M * 0.50;
    if (z < z_min) z = z_min;
    double ratio = DP_ROTOR_RADIUS_M / (4.0 * z);
    double factor = 1.0 / (1.0 - ratio * ratio);
    return clampd(factor, 0.85, 1.6);
}

/* Default is ANALYTICAL, not ML. With the quadratic actuator mapping
 * compensated in fc_step(), the corrected 8-seed closed-loop rerun still
 * gives analytical lower AGL RMS than the ML hybrid. Prediction error also
 * remains coupled to next-tick clearance, but the physically consistent
 * direction is overprediction -> too little commanded throttle -> clearance
 * loss (underprediction produces excess thrust, not a thrust deficit). The
 * ML branch retains its low-clearance analytical cutover as a safety guard;
 * pass GE_FF_MODE=ml explicitly to use the hybrid. */
ge_ff_mode_t ge_ff_mode_from_env(void) {
    const char *v = getenv("GE_FF_MODE");
    if (!v) return GE_FF_MODE_ANALYTICAL;
    if (!strcasecmp(v, "none") || !strcasecmp(v, "off") || !strcasecmp(v, "disabled")) return GE_FF_MODE_NONE;
    if (!strcasecmp(v, "analytical") || !strcasecmp(v, "cheeseman")) return GE_FF_MODE_ANALYTICAL;
    if (!strcasecmp(v, "ml") || !strcasecmp(v, "ai")) return GE_FF_MODE_ML;
    if (!strcasecmp(v, "ml_pure") || !strcasecmp(v, "pure_ml")) return GE_FF_MODE_ML_PURE;
    if (!strcasecmp(v, "ml_narrow") || !strcasecmp(v, "narrowed")) return GE_FF_MODE_ML_NARROW;
    if (!strcasecmp(v, "oracle")) return GE_FF_MODE_ORACLE;
    if (!strcasecmp(v, "ml_direct") || !strcasecmp(v, "pure_mlp")) return GE_FF_MODE_ML_DIRECT;
    if (!strcasecmp(v, "ml_asym_clamp") || !strcasecmp(v, "asym_clamp")) return GE_FF_MODE_ML_ASYM_CLAMP;
    return GE_FF_MODE_ANALYTICAL;
}

/* index 3 = rangefinder_alt in the feature ordering documented in
 * ge_ff_ml_weights.h -- keep in sync with the caller in fc_step(). */
#define GE_FF_ML_RANGEFINDER_IDX 3

double ge_ff_ml_predict(const double features[GE_FF_ML_N_FEATURES]) {
    if (isnan(features[GE_FF_ML_RANGEFINDER_IDX])) return NAN;

    double x[GE_FF_ML_N_FEATURES];
    for (int i = 0; i < GE_FF_ML_N_FEATURES; i++)
        x[i] = (features[i] - GE_FF_ML_SCALER_MEAN[i]) / GE_FF_ML_SCALER_SCALE[i];

    double h1[GE_FF_ML_H1];
    for (int j = 0; j < GE_FF_ML_H1; j++) {
        double s = GE_FF_ML_B0[j];
        for (int i = 0; i < GE_FF_ML_N_FEATURES; i++) s += GE_FF_ML_W0[j][i] * x[i];
        h1[j] = s > 0.0 ? s : 0.0; /* ReLU */
    }

    double h2[GE_FF_ML_H2];
    for (int j = 0; j < GE_FF_ML_H2; j++) {
        double s = GE_FF_ML_B1[j];
        for (int i = 0; i < GE_FF_ML_H1; i++) s += GE_FF_ML_W1[j][i] * h1[i];
        h2[j] = s > 0.0 ? s : 0.0; /* ReLU */
    }

    double out = GE_FF_ML_B2[0];
    for (int i = 0; i < GE_FF_ML_H2; i++) out += GE_FF_ML_W2[0][i] * h2[i];
    out += GE_FF_ML_BIAS_CORRECTION;

    /* Same physically-sane bounds the ground-truth ge_compute() and the
     * analytical estimate are both clamped to -- the network was only
     * ever trained on labels inside this range, so treat anything outside
     * it as extrapolation, not a real prediction. */
    return clampd(out, 0.85, 1.6);
}

ge_ff_source_t ge_ff_select(double analytical, double ai_pred, double band_frac, double *out_factor) {
    double lo = analytical * (1.0 - band_frac);
    double hi = analytical * (1.0 + band_frac);
    if (ai_pred >= lo && ai_pred <= hi) {
        *out_factor = ai_pred;
        return GE_FF_SRC_AI;
    }
    *out_factor = 1.0;
    return GE_FF_SRC_DISABLED;
}

double ge_ff_compensate_throttle(double throttle_raw, double factor) {
    if (!isfinite(factor) || factor <= 0.0) return NAN;
    return throttle_raw / sqrt(factor);
}

double ge_ff_asym_clamp_delta_from_env(void) {
    const char *v = getenv("GE_ASYM_CLAMP_DELTA");
    return v ? strtod(v, NULL) : 0.02;
}
