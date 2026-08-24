#include "mixer.h"
#include "drone_params.h"

static double clamp01(double v) {
    if (v < 0.0) return 0.0;
    if (v > 1.0) return 1.0;
    return v;
}

/* Signs derived directly from r x F with F = (0,0,thrust) per motor:
 * torque_x = +sum(y_i * T_i), torque_y = -sum(x_i * T_i). Reaction yaw
 * torque of a CW motor (dir=-1) pushes the frame the opposite (CCW, +z)
 * way, so its contribution to yaw_cmd is +, a CCW motor's is -. */
void mixer_compute(double throttle, double roll_cmd, double pitch_cmd, double yaw_cmd,
                    double out_cmd[4]) {
    for (int i = 0; i < 4; i++) {
        double nx = DP_MOTOR_LAYOUT[i].x / DP_ARM_LENGTH_M;
        double ny = DP_MOTOR_LAYOUT[i].y / DP_ARM_LENGTH_M;
        int dir = DP_MOTOR_LAYOUT[i].dir;

        double c = throttle
                 + roll_cmd  * ny
                 - pitch_cmd * nx
                 + yaw_cmd   * (-dir);
        out_cmd[i] = clamp01(c);
    }
}
