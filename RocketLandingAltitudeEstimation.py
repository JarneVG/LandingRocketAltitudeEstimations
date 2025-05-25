## Run with command: 'streamlit run RocketLandingAltitudeEstimation.py'

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

st.title("Rocket Altitude Simulation")

# === SIMULATION PARAMETERS ===
t_max = st.number_input("Total Simulation Time (s)", value=15.0)
dt = 0.001
n_steps = int(t_max / dt)


# === INPUT PARAMETERS ===
st.subheader("Select Rocket Parameters")
mass = st.number_input("Starting Rocket Mass (g)", value=595)
mass = mass/1000
drag_coefficient = st.number_input("Drag Coefficient", value=1.5)
rho = 1.17
diameter = st.number_input("Rocket Diameter (m)", value=0.066)
area = np.pi * diameter**2
g = 9.81







# === Thrust Curve Definitions ===
KlimaC6 = np.array([    
    [0, 0],
    [0.046, 0.953],
    [0.168, 5.259],
    [0.235, 10.023],
    [0.291, 15.00],
    [0.418, 9.87],
    [0.505, 7.546],
    [0.582, 6.631],
    [0.679, 6.136],
    [0.786, 5.716],
    [1.26, 5.678],
    [1.357, 5.488],
    [1.423, 4.992],
    [1.469, 4.116],
    [1.618, 1.22],
    [1.701, 0.0],
])

KlimaD3 = np.array([  
    [0, 0],
    [0.073, 0.229],
    [0.178, 0.686],
    [0.251, 1.287],
    [0.313, 2.203],
    [0.375, 3.633],
    [0.425, 5.006],
    [0.473, 6.465],
    [0.556, 8.181],
    [0.603, 9.01],
    [0.655, 6.922],
    [0.698, 5.463],
    [0.782, 4.291],
    [0.873, 3.576],
    [1.024, 3.146],
    [1.176, 2.946],
    [5.282, 2.918],
    [5.491, 2.832],
    [5.59, 2.517],
    [5.782, 1.859],
    [5.924, 1.287],
    [6.061, 0.715],
    [6.17, 0.286],
    [6.26, 0.0],
])

KlimaD9 = np.array([
    [0.000, 0.000],
    [0.040, 2.111],
    [0.116, 9.685],
    [0.213, 25.000],
    [0.286, 15.738],
    [0.329, 12.472],
    [0.369, 10.670],
    [0.420, 9.713],
    [0.495, 9.178],
    [0.597, 8.896],
    [1.711, 8.925],
    [1.826, 8.699],
    [1.917, 8.052],
    [1.975, 6.954],
    [2.206, 1.070],
    [2.242, 0.000],
])

motor_specs = {
    "Klima C6": {"propellant_mass": 0.0096, "burn_time": KlimaC6[-1, 0], "total_motor_mass": 0.0205},
    "Klima D3": {"propellant_mass": 0.017, "burn_time": KlimaD3[-1, 0], "total_motor_mass": 0.0279},
    "Klima D9": {"propellant_mass": 0.0161, "burn_time": KlimaD9[-1, 0], "total_motor_mass": 0.0271},
}


col1, col2 = st.columns(2)

with col1:
    st.subheader("Ascent")
    numOfMotors = st.slider("Number of Ascent Motors", 1, 5, value=2)
    motor_choice = st.selectbox("Select Motor for Ascent", ["Klima D9", "Klima D3", "Klima C6"])

    if motor_choice == "Klima C6":
        thrust_data = KlimaC6
    elif motor_choice == "Klima D3":
        thrust_data = KlimaD3
    elif motor_choice == "Klima D9":
        thrust_data = KlimaD9

    ejectAscentMotor = st.checkbox("Eject Ascent Motor At Burnout", value=True)


ejected_motor_mass = motor_specs[motor_choice]["total_motor_mass"] - motor_specs[motor_choice]["propellant_mass"]
ejected_motor_mass = ejected_motor_mass* numOfMotors
ascent_burn_time = motor_specs[motor_choice]["burn_time"]
ascent_propellant_mass = motor_specs[motor_choice]["propellant_mass"] * numOfMotors


with col2:
    st.subheader("Landing")
    numOfDescentMotors = st.slider("Number of Descent Motors", 1, 5, value=2)
    landing_motor_choice = st.selectbox("Select Landing Motor", ["Klima D3", "Klima D9", "Klima C6", "None"])

    # Define landing motor thrust curve
    if landing_motor_choice == "Klima C6":
        landing_thrust_data = KlimaC6
        landing_burn_time = motor_specs["Klima C6"]["burn_time"]
        landing_propellant_mass = motor_specs["Klima C6"]["propellant_mass"] * numOfDescentMotors
    elif landing_motor_choice == "Klima D3":
        landing_thrust_data = KlimaD3
        landing_burn_time = motor_specs["Klima D3"]["burn_time"]
        landing_propellant_mass = motor_specs["Klima D3"]["propellant_mass"] * numOfDescentMotors
    elif landing_motor_choice == "Klima D9":
        landing_thrust_data = KlimaD9
        landing_burn_time = motor_specs["Klima D9"]["burn_time"]
        landing_propellant_mass = motor_specs["Klima D9"]["propellant_mass"] * numOfDescentMotors
    else:
        landing_thrust_data = np.array([[0, 0]])  # No thrust
        landing_burn_time = 0.0
        landing_propellant_mass = 0.0

st.subheader("Delay After Apogee to Fire Landing Motor (s)")
landing_motor_delay = st.slider("Delay", 0.0, 5.0, 1.0, step=0.01)

landing_thrust_func = interp1d(
    landing_thrust_data[:, 0], landing_thrust_data[:, 1], bounds_error=False, fill_value=0.0
)



# Interpolate thrust curve
times = thrust_data[:, 0]
thrusts = thrust_data[:, 1]
thrust_func = interp1d(times, thrusts, bounds_error=False, fill_value=0.0)

# === SIMULATION SETUP ===
velocity = 0.0
altitude = 0.0

time_array = np.zeros(n_steps)
altitude_array = np.zeros(n_steps)
velocity_array = np.zeros(n_steps)
thrust_array = np.zeros(n_steps)
mass_array = np.zeros(n_steps)

max_altitude = -np.inf
has_fired_landing_motor = False

remaining_ascent_fuel = ascent_propellant_mass
remaining_landing_fuel = landing_propellant_mass

ascent_motor_ejected = False
landing_motor_burnout = False
touchdown_velocity = 0
landing_motor_start_time = 1000


# === SIMULATION LOOP ===
for i in range(n_steps):
    t = i * dt
    
    # Decrease mass due to fuel burn
    if t <= ascent_burn_time and remaining_ascent_fuel > 0:
        burn_rate_ascent = ascent_propellant_mass / ascent_burn_time
        mass -= burn_rate_ascent * dt
        remaining_ascent_fuel -= burn_rate_ascent * dt
    
    # Eject ascent motor if burn just ended
    if ejectAscentMotor and not ascent_motor_ejected and t > ascent_burn_time:
        mass -= ejected_motor_mass
        ascent_motor_ejected = True

    # Decrease mass during landing burn
    if has_fired_landing_motor and (t - landing_motor_start_time) <= landing_burn_time and remaining_landing_fuel > 0:
        burn_rate_landing = landing_propellant_mass / landing_burn_time
        mass -= burn_rate_landing * dt
        remaining_landing_fuel -= burn_rate_landing * dt
    
    # Check if motor has burned out
    if (t - landing_motor_start_time) >= landing_burn_time-0.5:
        landing_motor_burnout = True


    # Thrust from ascent motor
    ascent_thrust = thrust_func(t) * numOfMotors

    # Detect apogee
    if altitude > max_altitude:
        max_altitude = altitude
        apogee_time = t

    # Check landing motor condition
    if apogee_time is not None and not has_fired_landing_motor:
        if t >= apogee_time + landing_motor_delay:
            landing_motor_start_time = t
            has_fired_landing_motor = True

    landing_thrust = 0.0
    if has_fired_landing_motor:
        # Time since landing motor started
        t_landing = t - landing_motor_start_time
        landing_thrust = numOfDescentMotors*landing_thrust_func(t_landing)

    total_thrust = ascent_thrust + landing_thrust

    weight = mass * g
    drag = 0.5 * area * rho * drag_coefficient * velocity**2 * np.sign(velocity)

    net_force = total_thrust - drag - weight
    acceleration = net_force / mass

    # Prevent negative acceleration before launch
    if acceleration < 0 and t < 1:
        acceleration = 0

    velocity += acceleration * dt
    altitude += velocity * dt

    time_array[i] = t
    altitude_array[i] = altitude
    velocity_array[i] = velocity
    thrust_array[i] = total_thrust
    mass_array[i] = mass

    if altitude < 0:
        altitude_array[i:] = 0
        touchdown_velocity = velocity
        break
    
    


# only keep the time array up to the last valid altitude
# delete last mass point
time_array = time_array[:i]
altitude_array = altitude_array[:i]
thrust_array = thrust_array[:i]
mass_array = mass_array[:i]

# === PLOTTING ===
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(time_array, altitude_array, label='Altitude (m)', color='b')
ax.plot(time_array, thrust_array, '--', label='Thrust (N)', color='r')
ax.set_xlabel("Time (s)")
ax.set_ylabel("Altitude / Thrust")
ax.set_title("Rocket Altitude and Thrust Profile")
ax.legend()
ax.grid(True)

drop_height = (touchdown_velocity ** 2) / (2 * g)
# Compose annotation text without empty lines
text_lines = []

if not landing_motor_burnout:
    text_lines.append("Motor still firing at touchdown!")

text_lines.append(f"Equivalent drop height: {drop_height:.2f} m")

textstr = '\n'.join(text_lines)

props = dict(boxstyle='round', facecolor='white', alpha=0.8)
ax.text(0.99, 0.8, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right', bbox=props)


st.pyplot(fig)


if drop_height < 0.5 and i < n_steps-5 and landing_motor_burnout == True:
    st.balloons()

# === MASS OR WEIGHT PLOT ===
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(time_array, mass_array, label='Mass (kg)', color='g')
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Mass (kg)")
ax2.set_title("Rocket Mass Over Time")
ax2.legend()
ax2.grid(True)

st.pyplot(fig2)

