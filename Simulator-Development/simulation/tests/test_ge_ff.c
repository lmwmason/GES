#include "ge_ff.h"
#include <math.h>
#include <stdio.h>

static int near(double a, double b) { return fabs(a - b) < 1e-12; }

int main(void) {
    if (!near(ge_ff_compensate_throttle(0.8, 1.0), 0.8)) return 1;
    if (!near(ge_ff_compensate_throttle(0.8, 1.44), 0.8 / 1.2)) return 2;
    if (!near(pow(ge_ff_compensate_throttle(0.8, 1.44), 2.0) * 1.44,
              0.8 * 0.8)) return 3;
    if (!isnan(ge_ff_compensate_throttle(0.8, 0.0))) return 4;
    if (!isnan(ge_ff_compensate_throttle(0.8, NAN))) return 5;
    puts("sqrt actuator mapping preserves requested thrust");
    return 0;
}
