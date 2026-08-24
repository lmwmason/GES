#ifndef FW_PID_H
#define FW_PID_H

typedef struct {
    double kp, ki, kd;
    double integ;
    double prev_err;
    double integ_limit;
    double out_limit;
    double deriv_filt;
    double deriv_tau; /* low-pass time constant applied to the D term, seconds */
    int has_prev;
} pid_t;

/* deriv_tau smooths the derivative term before it's applied -- a raw
 * derivative of a noisy sensor sampled at 1kHz amplifies that noise by
 * ~1/dt, which is enough to make the output chatter between its limits.
 * Pass 0 to disable filtering (fine for already-clean signals like gyro). */
void pid_init(pid_t *p, double kp, double ki, double kd, double integ_limit,
              double out_limit, double deriv_tau);
void pid_reset(pid_t *p);
double pid_update(pid_t *p, double err, double dt);

#endif
