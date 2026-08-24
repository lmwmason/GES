# GES

![C](https://img.shields.io/badge/C-00599C?style=flat&logo=c&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-064F8C?style=flat&logo=cmake&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat&logo=python&logoColor=white)
![PyQt5](https://img.shields.io/badge/PyQt5-41CD52?style=flat&logo=qt&logoColor=white)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)

**GES: analyzing drone movement induced by ground effect, and correcting it with machine learning.**

When a multirotor flies close to the ground, rotor downwash reflecting off the surface changes
thrust in ways a flight controller doesn't normally account for, producing unwanted altitude and
attitude disturbances during low-altitude flight and landing. This project simulates that
phenomenon with a custom C physics engine, and tests whether ground effect can be predicted and
compensated for in real time using only sensors a real drone can carry (gyro, downward
rangefinder, motor current/RPM, commanded thrust) — evaluated both offline and by wiring the
trained model into the actual flight controller and flying it in closed-loop simulation.

## Repository layout

```
Drone-IMU-Ground-Sensing-GE/
├── LICENSE
├── references/                Prior-work papers (ground-effect physics, learning-based aero correction, related applications)
├── pre-results/                Superseded/invalidated experiment outputs, kept for the record
└── Simulator-Development/simulation/
    ├── physics/                Physics engine: rigid-body dynamics, motor model, terrain, ground-effect ground truth
    ├── firmware/                Flight controller (C): PID loops, mixer, analytical/ML feedforward
    ├── sim/                     Main simulation loop + CSV/UDP telemetry
    ├── common/, viz/            Shared math utilities + real-time 3D viewer (PyQt/OpenGL)
    ├── dataset/                 Generated training datasets (CSV)
    └── ml/
        ├── data.py, models_torch.py, train_*.py     Training pipeline (classical models + LSTM/Transformer)
        ├── export_mlp_to_c.py                        Exports a trained MLP into a C header for the firmware
        ├── analysis/                                 Result analysis / plotting scripts
        └── results/                                   Per-experiment outputs (plots, CSVs, model weights)
```

## Getting started

```bash
cd Simulator-Development/simulation

# Build and run the simulator (C, CMake)
./run.sh                              # build + run
./run.sh --viz                        # + real-time 3D viewer

# Generate a dataset
./generate_dataset.sh dataset/my_run 100 60 1   # 100 episodes, 60s each, starting at seed 1

# ML pipeline
python3 -m venv .venv && .venv/bin/pip install -r ml/requirements.txt
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu   # needed for the deep-learning models
.venv/bin/python3 ml/train_all_models.py --csv dataset/dataset_all.csv --lead-s 0.1

# Closed-loop feedforward comparison (none / analytical / ml modes, IMU + altitude tracking)
.venv/bin/python3 ml/analysis/ff_mode_comparison.py --seeds 1,2,3,4,5,6,7,8
```

The flight controller selects its ground-effect feedforward strategy via the `GE_FF_MODE`
environment variable (`none` / `analytical` / `ml`, see `firmware/ge_ff.c`), so the same compiled
firmware can be A/B/C-tested against itself.

## References

Prior work collected in `references/` — see each PDF for full text:

- Cheeseman & Bennett (1955), *The Effect of the Ground on a Helicopter Rotor in Forward Flight*, ARC R&M No. 3021
- Hooi, Lagor & Paley (2015), *Flow Sensing, Estimation and Control for Rotorcraft in Ground Effect*, IEEE Aerospace Conference
- Sanchez-Cuevas, Heredia & Ollero (2017), *Characterization of the Aerodynamic Ground Effect and Its Influence in Multirotor Control*, Journal of Sensors
- Shi et al. (2019), *Neural Lander: Stable Drone Landing Control Using Learned Dynamics*, ICRA
- Shi et al. (2021), *Neural-Swarm2: Planning and Control of Heterogeneous Multirotor Swarms using Learned Interactions*, IEEE T-RO
- O'Connell et al. (2022), *Neural-Fly Enables Rapid Learning for Agile Flight in Strong Winds*, Science Robotics
- Santos, Saska & Nascimento (2024), *A Generalized Thrust Estimation and Control Approach for Multirotors Micro Aerial Vehicles*, IEEE RA-L
- Yang et al. (2025), *Ground-Effect-Aware Modeling and Control for Multicopters*, arXiv:2506.19424
- Zou et al. (2023), *Perch a Quadrotor on Planes by the Ceiling Effect*, IROS
- Gräfe et al. (2026), *How to Model Your Crazyflie Brushless*, ICRA

## License

See [LICENSE](LICENSE).
