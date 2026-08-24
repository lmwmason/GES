#ifndef MIXER_H
#define MIXER_H

/* X-quad mixer. roll/pitch/yaw are small normalized torque-request terms
 * (same convention as motor_cmd, roughly -1..1 but expected to stay well
 * inside that range so throttle + trims never saturate hard). */
void mixer_compute(double throttle, double roll_cmd, double pitch_cmd, double yaw_cmd,
                    double out_cmd[4]);

#endif
