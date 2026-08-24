#include "pid.h"

static double clamp_local(double v, double lo, double hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

void pid_init(pid_t *p, double kp, double ki, double kd, double integ_limit,
              double out_limit, double deriv_tau) {
    p->kp = kp; p->ki = ki; p->kd = kd;
    p->integ_limit = integ_limit;
    p->out_limit = out_limit;
    p->deriv_tau = deriv_tau;
    pid_reset(p);
}

void pid_reset(pid_t *p) {
    p->integ = 0.0;
    p->prev_err = 0.0;
    p->deriv_filt = 0.0;
    p->has_prev = 0;
}

double pid_update(pid_t *p, double err, double dt) {
    if (dt <= 0.0) return 0.0;
    p->integ = clamp_local(p->integ + err * dt, -p->integ_limit, p->integ_limit);

    double raw_deriv = p->has_prev ? (err - p->prev_err) / dt : 0.0;
    p->prev_err = err;
    p->has_prev = 1;

    if (p->deriv_tau > 0.0) {
        double alpha = dt / (p->deriv_tau + dt);
        p->deriv_filt += alpha * (raw_deriv - p->deriv_filt);
    } else {
        p->deriv_filt = raw_deriv;
    }

    double out = p->kp * err + p->ki * p->integ + p->kd * p->deriv_filt;
    return clamp_local(out, -p->out_limit, p->out_limit);
}
