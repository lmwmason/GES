#ifndef FC_API_H
#define FC_API_H

#include "sensor_types.h"

/* Public ABI of the "external firmware" shared object (libfc.so). The
 * simulator dlopen()s this file and calls it purely through dlsym'd
 * function pointers -- exactly as if it were flashing/loading a separate
 * firmware image onto a flight controller, never linking against it. */

#ifdef __cplusplus
extern "C" {
#endif

void fc_init(void);
void fc_step(const sensor_packet_t *in, actuator_packet_t *out);

/* Firmware-internal ground-effect feedforward state, exposed purely for
 * telemetry/analysis. Source values:
 *   0 = analytical (flat-ground, rangefinder-based) compensation applied
 *   1 = AI-predicted compensation applied
 *   2 = disabled (AI disagreed with analytical baseline by more than the
 *       trust band, so no feedforward was applied this tick) */
typedef struct {
    double ge_ff_analytical;
    double ge_ff_ai;
    double ge_ff_applied;
    double ge_oracle_truth;
    double ge_oracle_bias;
    int    ge_ff_source;
    int    ge_ff_clamped;
    double thrust_cmd_norm;
} fc_debug_t;

void fc_get_debug(fc_debug_t *out);

/* Diagnostic-only simulator hook. Ground truth never enters sensor_packet_t
 * and is unavailable on real hardware. The simulator calls this immediately
 * before fc_step() when an oracle experiment is requested. */
void fc_set_diagnostic_ge_truth(double factor, double signed_bias);
void fc_set_diagnostic_prediction_bias(double signed_bias);

#ifdef __cplusplus
}
#endif

#endif
