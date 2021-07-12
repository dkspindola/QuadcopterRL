import numpy as np

config = {
    "env": "Quadcopter",
    "env_config": {
        # time step [s]
        "dt": 0.005,

        # gravitational acceleration in world coordinate system [m/s^2]
        "grav_acc": np.array([0, 0, -9.81]),

        # mass of quadcopter [kg]
        "standard_mass": 0.028,
        # distance from quadcopter center to propeller center [m]
        "standard_arm_length": 0.04,
        # motor 2% settling time [s]
        "standard_settling_time": 0.15,
        # thrust to weight ratio []
        "standard_thrust_to_weight": 1.9,
        # thrust to torque coefficient []
        "standard_thrust_to_torque": 0.006,

        # maximal positional error to end position in world coordinate system [m]
        "max_pos_err": np.array([2, 2, 2]),
        # maximal linear velocity in any direction in world coordinate system [m/s]
        "max_lin_vel": np.array([1, 1, 1]),
        # maximal angles in body coordinate system to sample from [rad]
        "max_ang": np.array([np.pi, np.pi, np.pi]),
        # maximal angular velocity around any axis in body coordinate system [rad/s]
        "max_ang_vel": np.array([2 * np.pi, 2 * np.pi, 2 * np.pi]),

        # non negative scalar weights (pos_err, lin_vel_err, rot_ang, ang_vel_err, action) for reward/cost function
        "weights": np.array([1, 0, 0, 0.1, 0.05]),
    },
    "model": {
       "fcnet_hiddens": [64, 64],
    },
    "num_workers": 2,
    "num_cpus_per_worker": 3,
    "framework": "torch",
    "horizon": 800,
    "no_done_at_end": True,
}

stop = {
}
