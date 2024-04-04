import numpy as np
import control as ctrl
import matplotlib.pyplot as plt

np.set_printoptions(precision=12)
np.set_printoptions(suppress=True)

# Constant parameters (all in SI units, except for motor_Kv, which is in rpm/V)
# Have a look...
"""
Quadcopter Utilities
"""

# Quadcopter parameters and dynamics
# You don't need to modify anything in this block of code - just run it and read
# the documentation of all functions to understand how to use them.

# Constant parameters (all in SI units, except for motor_Kv, which is in rpm/V)
# Dictionary
quadcopter_parameters = {
    'num_motors': 4,
    'quadrotor_mass': 1.8,
    'arm_length': 22.5/100,
    'air_density': 1.225,
    'gravity': 9.81,
    'moi_xx': 0.01788,
    'moi_yy': 0.03014,
    'moi_zz': 0.04614,
    'motor_Kv': 1000,
    'motor_time_constant': 50/1000,
    'rotor_mass': 40 / 1000,
    'rotor_radius': 19 / 1000,
    'motor_mass': 112 / 1000,
    'thrust_coeff': 0.112,
    'power_coeff': 0.044,
    'propeller_mass': 9 / 1000,
    'propeller_diameter': 10 * 0.0254,
    'voltage_max': 16.8,
    'voltage_min': 15
    }

def derived_quadcopter_parameters(quad_params):
    """
    This function updates the dictionary of quadcopter parameters with some
    derived parameters, such as the moment of inertia of the motors and the
    propellers, the hovering spin and more.

    Computes the parameters: k1, k2, (k3_x, k3_y, k3_z), k4_z and Γ_n, Γ_u.

    :param quad_params: dictionary of quadcopter parameters
    :returns: updated quadcopter parameters with extra KV pairs
    """
    motor_moi = quad_params['rotor_mass'] * quad_params['rotor_radius'] ** 2
    propeller_moi = (quad_params['propeller_mass'] * quad_params['propeller_diameter']**2) / 12
    hovering_spin = np.sqrt((quad_params['quadrotor_mass'] * quad_params['gravity'])
        / (quad_params['num_motors'] * quad_params['thrust_coeff']
            * quad_params['air_density']
            * (quad_params['propeller_diameter'] ** 4)))
    quad_params['k1'] = (quad_params['motor_Kv']
                        * (quad_params['voltage_max'] - quad_params['voltage_min'])) / 60
    quad_params['k2'] = 1 / quad_params['motor_time_constant']
    quad_params['k3_x'] = (2 * hovering_spin * quad_params['thrust_coeff']
                            * quad_params['air_density']
                            * (quad_params['propeller_diameter'] ** 4)
                            * quad_params['num_motors'] * quad_params['arm_length']) / ((2 ** 0.5) * quad_params['moi_xx'])
    quad_params['k3_y'] = (2 * hovering_spin * quad_params['thrust_coeff']
                            * quad_params['air_density']
                            * (quad_params['propeller_diameter'] ** 4)
                            * quad_params['num_motors']
                            * quad_params['arm_length']) / ((2 ** 0.5) * quad_params['moi_yy'])
    quad_params['k3_z'] = (2 * hovering_spin * quad_params['power_coeff']
                            * quad_params['air_density']
                            * (quad_params['propeller_diameter'] ** 5)
            * quad_params['num_motors']) / (2 * np.pi * quad_params['moi_zz'])
    quad_params['k4_z'] = (2 * np.pi * quad_params['num_motors']
                            * (propeller_moi + motor_moi)) / quad_params['moi_zz']
    quad_params['gamma_n'] = np.diagflat([quad_params['k3_x'],
                                            quad_params['k3_y'],
                                            quad_params['k3_z']
                                        - (quad_params['k4_z'] * quad_params['k2'])])
    quad_params['gamma_u'] = np.diagflat([0, 0, quad_params['k4_z']
                                        * quad_params['k2']
                                        * quad_params['k1']])
    return quad_params


def system_matrices(quad_params):
    """
    This function returns the matrices A, B and C of the continuous-time linear
    dynamical system:
    dx(t)/dt = A_c x(t) + B_c u(t)
    y(t) = C_c x(t)

    The system state, x, is described above.

    Usage:
    a_c, b_c, c_c = system_matrices(quad_params)

    :param quad_params: dictionary with quadcopter parameters
    :returns: continuous-time system matrices, A, B and C
    """
    a_c = np.zeros(shape=(9, 9))
    b_c = np.zeros(shape=(9, 3))
    c_c = np.zeros(shape=(6, 9))
    a_c[0:3, 3:6] = 0.5 * np.eye(3)
    a_c[3:6, 6:9] = quad_params['gamma_n']
    a_c[6:9, 6:9] = -quad_params['k2']*np.eye(3)
    b_c[3:6, 0:3] = quad_params['gamma_u']
    b_c[6:9, 0:3] = quad_params['k2'] * quad_params['k1'] * np.eye(3)
    c_c[0:6, 0:6] = np.eye(6)
    return a_c, b_c, c_c


def continuous_time_system_matrices(quad_params):
    """
    This function returns the matrices A_c, B_c and C_c of the continuous-time
    linear system

    x' = A_c x + B_c u
    y = C x

    :param quad_params: quadcopter parameters
    :returns: matrices a_c, b_c, c_c (as numpy arrays)
    """
    quad_params = derived_quadcopter_parameters(quad_params)
    a_c, b_c, c_c = system_matrices(quad_params)
    return a_c, b_c, c_c


def angles_from_x(x):
    """
    This function takes the state, x, of the quadcopter and returns the three
    Euler angles, φ, θ and ψ, in an numpy array. The Euler angles are in degrees.

    :param x: state vector (numpy array)
    :returns: Euler angles (numpy array)
    """
    q0 = np.sqrt(1 - x[0]**2 - x[1]**2 - x[2]**2)
    q = np.array([q0, x[0], x[1], x[2]])
    phi = np.arctan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))
    theta = np.arcsin(2 * (q[0] * q[2] - q[1] * q[3]))
    psi = np.arctan2(2 * (q[0] * q[3] + q[2] * q[1]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
    return np.rad2deg(np.array([phi, theta, psi]))

def angles_to_quaternion(phi_deg, theta_deg, psi_deg):
    """
    This function takes the three Euler angles and returns the corresponding
    rotation quaternion as a numpy array.

    :param phi_deg: roll angle in degrees
    :param theta_deg: pitch angle in degrees
    :param psi_deg: yaw angle in degrees
    :returns: quaternion as numpy array
    """
    phi = np.deg2rad(phi_deg)
    theta = np.deg2rad(theta_deg)
    psi = np.deg2rad(psi_deg)
    cphi = np.cos(phi/2)
    ctheta = np.cos(theta/2)
    cpsi = np.cos(psi/2)
    sphi = np.sin(phi/2)
    stheta = np.sin(theta/2)
    spsi = np.sin(psi/2)
    return np.array([cphi*ctheta*cpsi + sphi*stheta*spsi,
                    sphi*ctheta*cpsi - cphi*stheta*spsi,
                    cphi*stheta*cpsi + sphi*ctheta*spsi,
                    cphi*ctheta*spsi - sphi*stheta*cpsi])

"""""
# <<<TASK 4>>>
# Obtain the discrete-time matrices of this system (namely, a_d, b_d, and c_d)
# and check whether the system is controllable and observable [use ctrl.ctrb and
# ctrl.obsv]
# NOTE: the sampling time is 1/150 s.
# INCLUDE THIS IN YOUR FINAL NOTEBOOK OR REPORT
"""""
quad_params = derived_quadcopter_parameters(quadcopter_parameters)
# a_c, b_c, c_c = system_matrices(quad_params)
t_sampling = 1/150
a_c, b_c, c_c =  continuous_time_system_matrices(quad_params)

continuous_system = ctrl.ss(a_c, b_c, c_c, 0)


# Use ctrb.c2d to discretise the continuous-time dynamical system; use the
# above sampling time
discrete_time_system = ctrl.c2d(continuous_system ,t_sampling)
a_d = discrete_time_system.A
b_d = discrete_time_system.B
c_d = discrete_time_system.C


# Compute the controllability matrix of the discrete-time system
controllability_matrix = ctrl.ctrb(a_d, b_d)

# this matirx must have RANK = 9 (be FULL RANK)
ctrl_matrix_rank = np.linalg.matrix_rank(controllability_matrix)
# print(ctrl_matrix_rank)

# TASK FOR YOU...
# CHECK WHETHER THE (discrete-time) SYSTEM IS OBSERVABLE
# Sploilers: YES!
observability_matrix = ctrl.obsv(a_d, c_d)
obsv_matrix_rank = np.linalg.matrix_rank(observability_matrix)

# Use the above matrices Q and R as a starting point (you may modify them to
# tune the system to your liking) [use np.diagflat]
q_lqr = np.diagflat([1500, 1500, 10000, 2, 2, 10, 1, 1, 1])
r_lqr = np.diagflat([1, 1, 10])

"""""
# <<<TASK 5>>>
# Use ctrl.dare to compute the LQR gain with the above weight matrices
# What are the eigenvalues of the closed-loop system matrix, A + BK?
# INCLUDE THIS IN YOUR FINAL NOTEBOOK OR REPORT
"""""
# Use
P_lqr, eigenvalues_cl, neg_lqr_gain = ctrl.dare(a_d, b_d, q_lqr, r_lqr)

lqr_gain = -neg_lqr_gain

# LQR simulations
P0 = np.diagflat([50, 50, 50, 10, 10, 10, 100, 100, 100])
Q_Kf = np.diagflat([0.5, 0.5, 0.5, 1.2, 1.2, 1.2, 2, 2, 2]) / 1e8
R_Kf = np.diagflat([0.8, 0.8, 0.8, 1, 1, 1]) / 1e5

cl_matrix = a_d + b_d @ lqr_gain
t_sim = 1000

# Initialise cache
angles_cache = np.zeros(shape=(t_sim + 1, 3))
quat_init = angles_to_quaternion(2, -2, 6)
x = np.array([quat_init[1], quat_init[2], quat_init[3], 0, 0, 0, 0, 0, 0])
angles = angles_from_x(x)
angles_cache[0, :] = angles

for t in range(t_sim):
    w = np.random.multivariate_normal(np.zeros(9), Q_Kf)
    # x_next = Ax + Bu, where u = Kx
    # so
    # x_next = (A+BK)x + w
    x = cl_matrix @ x + w
    angles = angles_from_x(x)
    angles_cache[t + 1, :] = angles
plt.figure(1)
plt.plot(np.arange(0, 200) / 150, angles_cache[0:200, 0:2])
plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Euler angles (deg)')
plt.legend(['pitch', 'roll'])

plt.figure(2)
plt.plot(np.arange(0, t_sim + 1) / 150, angles_cache[:, 2])
plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Yaw (deg)')
plt.show()
"""""
# <<<TASK 7>>>
# Here you need to combine the LQR from Task 6 with a Kalman filter. Start by
# implementing the following functions
"""""
Q_Kf = 1e-8 * np.diagflat([0.5, 0.5, 0.5, 1.2, 1.2, 1.2, 2, 2, 2])
R_Kf = 1e-5 * np.eye(6)

def state_update(x, u):
    """
    Computes Ax + Bu + w
    """
    w = np.random.multivariate_normal(np.zeros((9,)), Q_Kf)
    w = np.reshape(w, (9, 1))

    x_next = a_d @ x + b_d @ u + w
    return x_next

x = np.ones((9, 1))
u = np.array([[0.4], [0.5], [0.6]])

# print(state_update(x,u))

def output(x):
    """
    Computes y = Cx + v
    """
    v = np.random.multivariate_normal(np.zeros((6,)), R_Kf)
    y = c_d @ x + v

    return y

# print(output(x))

def measurement_update(y, z_pred, sigma_pred):
    """
    Measurement update of the Kalman filter
    Returns the corrected state and covariance estimates after the output
    is measured
    """
    F = c_d @ sigma_pred @ c_d.T + R
    output_error = y - c_d @ z_pred
    z_corrected = z_pred + sigma_pred @ c_d.T @ output_error / F[0, 0]

    sigma_corrected = sigma_pred - sigma_pred @ c_d.T @ (c_d @ sigma_pred) / F[0, 0]

    return z_corrected, sigma_corrected

# print(measurement_update(u , u , u))

def time_update(x_meas_update, u, sigma_meas_update):
    """
    Measurement update of the Kalman filter
    Don't forget the input!
    """
    x_predicted = a_d @ x_meas_update
    sigma_predicted = a_d @ sigma_meas_update @ a_d.T + Q_Kf * (G @ G.T)
    return x_predicted, sigma_predicted

# Once you implement the above functions, you can use them to write a for loop
# and simulate the closed-loop system.
# INCLUDE THIS IN YOUR FINAL NOTEBOOK OR REPORT