#include "aero_ground_effect.h"
#include <math.h>
#include <stdio.h>

int main(void) {
    const double r = 0.127;
    double cb = ge_flat_factor(2.0 * r, r, GE_BASE_CHEESEMAN);
    double hayden = ge_flat_factor(2.0 * r, r, GE_BASE_HAYDEN);
    if (!isfinite(cb) || !isfinite(hayden) || cb <= 1.0 || hayden <= 1.0) return 1;
    if (fabs(cb - hayden) < 1e-4) return 2;
    if (fabs(ge_flat_factor(100.0 * r, r, GE_BASE_HAYDEN) - 1.0) > 1e-12) return 3;
    puts("Cheeseman and Hayden ground-effect bases are finite and distinct");
    return 0;
}
