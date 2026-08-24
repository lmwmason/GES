#ifndef ATTITUDE_ESTIMATOR_H
#define ATTITUDE_ESTIMATOR_H

/* Complementary filter over gyro+accel only (no magnetometer -> yaw is
 * gyro-integrated and free-drifts, same limitation a real low-cost FC
 * without a compass would have). Body frame: x-forward, y-right, z-up. */
typedef struct {
    double roll, pitch, yaw; /* rad */
    int initialized;
} attitude_estimator_t;

void ae_init(attitude_estimator_t *ae);
void ae_update(attitude_estimator_t *ae, const double gyro[3], const double accel[3], double dt);

#endif
