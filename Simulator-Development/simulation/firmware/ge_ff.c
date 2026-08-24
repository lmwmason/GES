#include "ge_ff.h"
#include "ge_ff_ml_weights.h"
#include "drone_params.h"
#include "math3d.h"
#include <math.h>
#include <stdlib.h>
#include <strings.h>

double ge_ff_analytical_factor(double rangefinder_alt_m) {
    if (isnan(rangefinder_alt_m) || rangefinder_alt_m <= 0.0) return 1.0;
    double z = rangefinder_alt_m;
    double z_min = DP_ROTOR_RADIUS_M * 0.50;
    if (z < z_min) z = z_min;
    double ratio = DP_ROTOR_RADIUS_M / (4.0 * z);
    double factor = 1.0 / (1.0 - ratio * ratio);
    return clampd(factor, 0.85, 1.6);
}

/* Default is ANALYTICAL, not ML. docs/research_report.md Sections 5.9-5.11
 * found (and confirmed the mechanism for) a real offline-vs-closed-loop
 * failure mode: the trained MLP is far more accurate offline (Section
 * 5.2b), but under closed-loop control its error spikes right at the
 * lowest-clearance instants and that spike compounds into an even worse
 * clearance the next tick. GE_FF_MODE_ML's fc_step() branch now hard-cuts
 * over to the analytical estimate below GE_FF_LOW_CLEARANCE_M specifically
 * to break that feedback loop (Section 5.12), which brought it back down
 * to statistically match analytical's closed-loop performance -- but that
 * only confirms "no longer worse," not "meaningfully better" once the
 * dangerous zone is patched over with analytical anyway, so analytical
 * alone remains the default: it's simpler, has zero training cost, and
 * the ML hybrid's advantage in the non-danger-zone majority of flight
 * hasn't been shown to be statistically significant yet. Pass
 * GE_FF_MODE=ml explicitly to use the hybrid. */
ge_ff_mode_t ge_ff_mode_from_env(void) {
    const char *v = getenv("GE_FF_MODE");
    if (!v) return GE_FF_MODE_ANALYTICAL;
    if (!strcasecmp(v, "none") || !strcasecmp(v, "off") || !strcasecmp(v, "disabled")) return GE_FF_MODE_NONE;
    if (!strcasecmp(v, "analytical") || !strcasecmp(v, "cheeseman")) return GE_FF_MODE_ANALYTICAL;
    if (!strcasecmp(v, "ml") || !strcasecmp(v, "ai")) return GE_FF_MODE_ML;
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
