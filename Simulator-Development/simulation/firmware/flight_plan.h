#ifndef FLIGHT_PLAN_H
#define FLIGHT_PLAN_H

/* Scripted mission: takeoff to a low hover height, settle, then cruise
 * forward (feedforward pitch lean -- there is no GPS/optical-flow in this
 * airframe's sensor set, so forward speed is open-loop by design) low
 * enough over the terrain for ground effect to matter, then level back out.
 * All setpoints a real onboard flight plan / RC input would provide. */
typedef struct {
    double target_agl;   /* altitude-hold setpoint, rangefinder AGL [m] */
    double target_roll;  /* rad */
    double target_pitch; /* rad, negative pitches nose down (forward lean) */
    double target_yaw_rate; /* rad/s, 0 = hold heading */
} flight_setpoint_t;

flight_setpoint_t flight_plan_eval(double t);

#endif
