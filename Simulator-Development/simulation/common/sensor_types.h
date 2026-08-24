#ifndef SENSOR_TYPES_H
#define SENSOR_TYPES_H

/* This is the entire ABI the firmware is allowed to see. A real STM32 FC
 * only has access to what its sensors report -- in particular it has NO
 * direct knowledge of terrain shape. The physics engine's ground-truth
 * terrain/ground-effect state must never leak into this struct. */
typedef struct {
    double t;                  /* simulation time [s] */
    double dt;                 /* time since last call [s] */
    double gyro[3];             /* body rate [rad/s] (noisy) */
    double accel[3];            /* specific force, body frame [m/s^2] (noisy) */
    double rangefinder_alt;     /* downward rangefinder, ground clearance [m], NAN if out of range */
    double baro_alt;            /* barometric altitude rel. to takeoff [m] (noisy) */
    double motor_rpm_meas[4];   /* ESC telemetry RPM, order matches DP_MOTOR_LAYOUT */
    double motor_current[4];    /* per-motor current [A] */
    double battery_voltage;     /* [V] */
} sensor_packet_t;

typedef struct {
    double motor_cmd[4]; /* normalized throttle command 0..1 per motor */
} actuator_packet_t;

#endif
