import numpy as np
from MathUtils import calc_rot_matrix

config = {
    "env": "Quadcopter",
    "env_config": {
        # time step [s]
        "dt": 0.005,

        # gravitational acceleration in world coordinate system [m/s^2]
        "grav_acc": np.array([0, 0, -9.81]),

        # mass of quadcopter [kg]
        "mass": 0.03,

        # thrust to weight ratio []
        "thrust_to_weight": 1.9,
        # torque to thrust coefficient []
        "torque_to_thrust": 0.006,

        # matrix of inertia [kg*m^2]
        "inertia": 10e-6 * np.array([[16.5717, 0.8308,  0.7183],
                                     [0.8308, 16.6556,  1.8002],
                                     [0.7183,  1.8002, 29.2617]]),

        # distance from quadcopter center to propeller center[m]
        "arm_length": 0.065,

        # start position of quadcopter in world coordinate system [m]
        "start_pos": 2 * np.random.rand(3) + np.array([0, 0, 2]),
        # start linear velocity of quadcopter in world coordinate system [m/s]
        "start_lin_vel": np.random.rand(3),
        # start rotation matrix based on given angles [rad] from body to world coordinate system []
        "start_rot_matrix": calc_rot_matrix(2 * np.pi * np.random.rand(), 2 * np.pi * np.random.rand(),
                                            2 * np.pi * np.random.rand()),
        # start angular velocity in body coordinate system [rad/s]
        "start_ang_vel": 2 * np.pi * np.random.rand(3),

        # end/goal position of agent in world coordinate system [m]
        "end_pos": np.array([0, 0, 2]),
        # end/goal linear velocity of agent in world coordinate system [m/s]
        "end_lin_vel": np.zeros(3),
        # end/goal rotation matrix based on given angles [rad] from body to world coordinate system []
        "end_rot_matrix": calc_rot_matrix(0, 0, 0),
        # end/goal angular velocity of agent in the body coordinate system [rad/s]
        "end_ang_vel": np.zeros(3),

        # motor 2% settling time [s]
        "settling_time": 0.15,
        # start motor lag []
        "start_motor_lag": np.zeros(4),

        # maximal positional error to end position in world coordinate system [m]
        "max_pos_err": 2 * np.ones(3),
        # maximal linear velocity in any direction in world coordinate system [m/s]
        "max_lin_vel": np.ones(3),
        # maximal rotation matrix []
        "max_rot_matrix": np.ones((3, 3)),
        # maximal angular velocity around any axis in body coordinate system [rad/s]
        "max_ang_vel": np.array([2 * np.pi, 2 * np.pi, 2 * np.pi]),

        # non negative scalar weights (pos_err, lin_vel_err, rot_ang, ang_vel_err, action) for reward/cost function
        "weights": np.array([1, 0, 0, 0.1, 0.05]),
    },
    "model": {
       "fcnet_hiddens": [64, 64],
    },
    "num_workers": 1,
    "framework": "torch",
}

stop = {
}
