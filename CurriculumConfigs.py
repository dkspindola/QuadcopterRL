import numpy as np
from numpy import pi
from Interval import Interval

# randomization of start position
env_config0 = {
    # maximal positional error to sample from [m]
    "max_pos_err": Interval(low=-2, high=2),
    # maximal linear velocity to sample from [m/s]
    "max_lin_vel": Interval(low=-1, high=1),
    # maximal angles to sample from [rad]
    "max_ang": Interval(low=0, high=0),
    # maximal angular velocity to sample from [rad/s]
    "max_ang_vel": Interval(low=-2 * pi, high=2 * pi),

    # percentage used to randomize quadcopter parameters
    "percentage": 0,

    "level": 0,
}

# randomization of start position, start linear velocity
env_config1 = {
    # maximal positional error to sample from [m]
    "max_pos_err": Interval(low=1, high=1),
    # maximal linear velocity to sample from [m/s]
    "max_lin_vel": Interval(low=-1, high=1),
    # maximal angles to sample from [rad]
    "max_ang": Interval(low=-pi/4, high=pi/4),
    # maximal angular velocity to sample from [rad/s]
    "max_ang_vel": Interval(low=-2 * pi, high=2 * pi),

    # percentage used to randomize quadcopter parameters
    "percentage": 0,

    "level": 1,
}

# randomization of start position, start linear velocity and start angular velocity
env_config2 = {
    # maximal positional error to sample from [m]
    "max_pos_err": Interval(low=-2, high=2),
    # maximal linear velocity to sample from [m/s]
    "max_lin_vel": Interval(low=-1, high=1),
    # maximal angles to sample from [rad]
    "max_ang": Interval(low=-pi/2, high=pi/2),
    # maximal angular velocity to sample from [rad/s]
    "max_ang_vel": Interval(low=-2 * pi, high=2 * pi),

    # percentage used to randomize quadcopter parameters
    "percentage": 0,

    "level": 2,
}

# randomization of start linear velocity, start angular velocity
env_config3 = {
    # maximal positional error to sample from [m]
    "max_pos_err": Interval(low=-2, high=2),
    # maximal linear velocity to sample from [m/s]
    "max_lin_vel": Interval(low=-1, high=1),
    # maximal angles to sample from [rad]
    "max_ang": Interval(low=- 3/4*pi, high=3/4*pi),
    # maximal angular velocity to sample from [rad/s]
    "max_ang_vel": Interval(low=-2 * pi, high=2 * pi),

    # percentage used to randomize quadcopter parameters
    "percentage": 0,

    "level": 3,
}

# randomization of start position, start linear velocity, start angular velocity and start angles
env_config4 = {
    # maximal positional error to sample from [m]
    "max_pos_err": Interval(low=-2, high=2),
    # maximal linear velocity to sample from [m/s]
    "max_lin_vel": Interval(low=-1, high=1),
    # maximal angles to sample from [rad]
    "max_ang": Interval(low=-pi, high=pi),
    # maximal angular velocity to sample from [rad/s]
    "max_ang_vel": Interval(low=-2 * pi, high=2 * pi),

    # percentage used to randomize quadcopter parameters
    "percentage": 0,

    "level": 4,
}
