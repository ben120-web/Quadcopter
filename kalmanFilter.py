import numpy as np
import matplotlib.pyplot as plt
"""""
# <<<TASK 1>>>
# Define the problem data (use the data provided in slides #14)
"""""
t_sampling = 0.05

# A = [1 h
#      0 1]
# G = [h
#      0]
A = np.array([[1, t_sampling],
             [0,1]])
G = np.array([[t_sampling],[0]])

# C = [1 0]

# We need C to be a row vector, that is, to have shape (1, 2)
# if we do: C = np.array([[1], [0]]), the shape will be (2, 1)
# if we do: C = np.array([1, 0]), the shape will be (2, )

C = np.array([[1,0]])
C = C.reshape(-1,2)

# P0 = [LARGE_VALUE     0
#       0               0]
P0 = np.array([[100,0],
               [0,0]])
Q = 8  # variance of w
R = 15  # variance of v

# INCLUDE THIS IN YOUR FINAL NOTEBOOK OR REPORT
"""""
# <<<TASK 2>>>
# Implement the following four functions (write documentation as well)
# The first one is given as an example
# INCLUDE THIS IN YOUR FINAL NOTEBOOK OR REPORT
"""""
def state_update(z):
    """
    This function takes the system state, z = (x, u_bar),
    where x is the position of our vehicle and u_bar is
    its velocity (which is not random), and returns the
    next state, according to:

    z_next = A*z + G*w

    :param z: system state
    :returns: next state
    """
    w = np.random.normal(0, Q)
    # compute and return Az + Gw
    z_next = A @ z + w * G
    return z_next


def output(z):
    """
    Given the system state, z, this function returns the output
    of the system, namely

    y = C*z + v,

    where v is an additive noise term.

    :param z: system state
    :return: system output
    """
    v = np.random.normal(0, R)
    y = C @ z + v
    return y[0, 0]


def measurement_update(y, z_pred, sigma_pred):
    """
    Implement the measurement update step
    """
    F = C @ sigma_pred @ C.T + R
    output_error = y - C @ z_pred
    z_corrected = z_pred + sigma_pred @ C.T @ output_error / F[0, 0]

    sigma_corrected = sigma_pred - sigma_pred @ C.T @ (C @ sigma_pred) / F[0, 0]

    return z_corrected, sigma_corrected

# Note:
# A <= B
# means that
# B - A is positive semidefinite
# equivalently:
# the eigenvalues of B - A are all nonnegative


def time_update(z_meas_update, sigma_meas_update):
    """
    Implement the time update step
    More documentation goes here

    :param z_meas_update: state from measurement update
    :param sigma_meas_update: variance from measurement update
    """
    z_predicted = A @ z_meas_update
    sigma_predicted = A @ sigma_meas_update @ A.T + Q * (G @ G.T)
    return z_predicted, sigma_predicted

# NOTE: Make sure the above functions work before you move on.

# Run simulations and record the time and measurement update steps
# of the Kalman filter
t_sim = 80
"""""
# TASK 3.1
# Simulate the system starting from a random initial state
"""""
z_actual_cache = np.zeros(shape=(t_sim + 1, 2))
z_corr_cache = np.zeros(shape=(t_sim, 2))
z_pred_cache = np.zeros(shape=(t_sim + 1, 2))
sigma_pred_cache = np.zeros(shape=(2, 2, t_sim + 1))  # tensor
sigma_corr_cache = np.zeros(shape=(2, 2, t_sim))

z_pred = np.array([[0], [10]])
sigma_pred = P0
# be careful: np.random.multivariate_normal needs
# the mean to be a (n, ) array
z_actual = np.random.multivariate_normal(z_pred[:, 0], sigma_pred)
z_actual = np.reshape(z_actual, (2, 1))
z_actual_cache[0, :] = z_actual.T
z_pred_cache[0, :] = z_pred.T  # <- don't forget the .T
sigma_pred_cache[:, :, 0] = P0

for t in range(t_sim):
    # --- OUTPUT MEASUREMENT
    # FLIP A "COIN" FIRST TO DECIDE WHETHER THERE IS AN OUTPUT
    # IF NOT, THE MEASUREMENT UPDATE CANNOT BE USED

    y = output(z_actual)

    # --- MEASUREMENT UPDATE
    z_corr, sigma_corr = measurement_update(y, z_pred, sigma_pred)
    z_corr_cache[t, :] = z_corr.T
    sigma_corr_cache[:, :, t] = sigma_corr
    # z_corr_cache = z_corr(t,:)
    # z_corr_cache[t, :] = z_corr

    # --- TIME UPDATE
    z_pred, sigma_pred = time_update(z_corr, sigma_corr)
    z_pred_cache[t+1, :] = z_pred.T
    sigma_pred_cache[:, :, t+1] = sigma_pred

    # --- STATE UPDATE
    z_actual = state_update(z_actual)
    z_actual_cache[t+1, :] = z_actual.T

time_scale = np.arange(0, t_sim + 1) * t_sampling
plt.plot(time_scale[0:t_sim], z_corr_cache[:, 0], label="Corrected")
plt.plot(time_scale, z_actual_cache[:, 0], label="Actual pos")
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.grid()
plt.legend()
plt.show()

"""""
# <TASK 3.2>
# Modify the code you produced in Task 3.1 and implement the 
# Kalman filter
"""""


"""""
# <TASK 3.3>
# Modify the code you produced in Task 3.2 and record the actual
# state of the system as well as the estimates produced by the 
# measurement and time update steps of the Kalman filter. Then,
# plot the estimated trajectories alongside the actual states.
"""""