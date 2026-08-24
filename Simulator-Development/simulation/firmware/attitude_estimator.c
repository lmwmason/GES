#include "attitude_estimator.h"
#include <math.h>

#define COMPL_ALPHA 0.98

void ae_init(attitude_estimator_t *ae) {
    ae->roll = 0.0;
    ae->pitch = 0.0;
    ae->yaw = 0.0;
    ae->initialized = 0;
}

void ae_update(attitude_estimator_t *ae, const double gyro[3], const double accel[3], double dt) {
    double roll_acc = atan2(accel[1], accel[2]);
    double pitch_acc = atan2(-accel[0], sqrt(accel[1] * accel[1] + accel[2] * accel[2]));

    if (!ae->initialized) {
        ae->roll = roll_acc;
        ae->pitch = pitch_acc;
        ae->yaw = 0.0;
        ae->initialized = 1;
        return;
    }

    double roll_gyro = ae->roll + gyro[0] * dt;
    double pitch_gyro = ae->pitch + gyro[1] * dt;

    ae->roll = COMPL_ALPHA * roll_gyro + (1.0 - COMPL_ALPHA) * roll_acc;
    ae->pitch = COMPL_ALPHA * pitch_gyro + (1.0 - COMPL_ALPHA) * pitch_acc;
    ae->yaw += gyro[2] * dt; /* no compass -> free-running integration */
}
