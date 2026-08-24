cmake_minimum_required(VERSION 3.16)
project(drone_ge_sim C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

add_compile_options(-Wall -Wextra -O2 -ffast-math)

include_directories(
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/physics
    ${CMAKE_SOURCE_DIR}/firmware
)

# ---- common math/types (static lib, linked by both physics and firmware) ----
add_library(common STATIC
    common/math3d.c
)
target_link_libraries(common m)

# ---- physics core (ground-truth rigid body + fluid/ground-effect model) ----
add_library(physics STATIC
    physics/terrain.c
    physics/aero_ground_effect.c
    physics/motor_model.c
    physics/rigid_body.c
)
target_link_libraries(physics common m)

# ---- external FC firmware, built as a standalone shared object ----
# This mirrors flashing a separate firmware image onto a real STM32 flight
# controller: the simulator only ever talks to it through fc_api.h and
# loads it dynamically (dlopen) at run time, never linking it directly.
add_library(fc SHARED
    firmware/attitude_estimator.c
    firmware/pid.c
    firmware/mixer.c
    firmware/flight_plan.c
    firmware/ge_ff.c
    firmware/controller.c
)
target_include_directories(fc PRIVATE ${CMAKE_SOURCE_DIR}/firmware)
target_link_libraries(fc common m)
set_target_properties(fc PROPERTIES PREFIX "lib" OUTPUT_NAME "fc")

# ---- simulator executable ----
add_executable(simulator
    sim/main.c
    sim/telemetry.c
)
target_link_libraries(simulator physics common m dl)
