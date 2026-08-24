#include "motor_model.h"
#include "drone_params.h"

void motor_init(motor_state_t *m) {
    m->rpm = 0.0;
}

void motor_update(motor_state_t *m, double cmd, double dt) {
    if (cmd < 0.0) cmd = 0.0;
    if (cmd > 1.0) cmd = 1.0;
    double target_rpm = cmd * DP_MOTOR_RPM_MAX;
    double alpha = dt / (DP_MOTOR_TAU_S + dt);
    m->rpm += alpha * (target_rpm - m->rpm);
}

double motor_thrust_N(double rpm) {
    return DP_MOTOR_KT * rpm * rpm;
}

double motor_torque_Nm(double rpm) {
    return DP_MOTOR_KQ * rpm * rpm;
}

double motor_current_A(double rpm) {
    double frac = rpm / DP_MOTOR_RPM_MAX;
    if (frac < 0.0) frac = 0.0;
    const double idle_a = 0.4;
    const double max_a = 22.0;
    return idle_a + max_a * frac * frac;
}
