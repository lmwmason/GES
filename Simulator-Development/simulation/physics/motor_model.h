#ifndef MOTOR_MODEL_H
#define MOTOR_MODEL_H

typedef struct {
    double rpm;
} motor_state_t;

void motor_init(motor_state_t *m);

/* cmd: normalized throttle 0..1. Drives m->rpm toward cmd*RPM_MAX with a
 * first-order lag (DP_MOTOR_TAU_S) approximating ESC+motor spin-up time. */
void motor_update(motor_state_t *m, double cmd, double dt);

double motor_thrust_N(double rpm);   /* free-air thrust, before any GE factor is applied */
double motor_torque_Nm(double rpm);  /* reaction torque magnitude about the rotor axis */
double motor_current_A(double rpm);  /* ESC current draw, function of rpm only */

#endif
