import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


# Streamlit App Title
st.title("Endangered Species Population Simulator with Environmental and Human Impacts")

# Sidebar for User Inputs
st.sidebar.header("Simulation Parameters")

# Population Parameters
st.sidebar.subheader("Population Parameters")
N0 = st.sidebar.number_input("Initial Population (N0)", value=300, min_value=10)
r = st.sidebar.slider("Reproduction Rate (r)", min_value=0.1, max_value=1.0, value=0.4, step=0.1)
K_max = st.sidebar.number_input("Maximum Carrying Capacity (K_max)", value=1000, min_value=100)

# Environmental Parameters
st.sidebar.subheader("Environmental Factors")
R_mean = st.sidebar.number_input("Mean Rainfall (mm)", value=1000, min_value=500, max_value=2000)
R_sd = st.sidebar.number_input("Rainfall Variability (± mm)", value=100, min_value=0, max_value=500)
T_initial = st.sidebar.number_input("Initial Temperature (°C)", value=20, min_value=-10, max_value=50)
T_rate = st.sidebar.slider("Temperature Rise Rate (°C/year)", min_value=0.0, max_value=0.1, value=0.02, step=0.01)
temp_sd = st.sidebar.number_input("Temperature Variability (°C)", value=1.0, min_value=0.0, max_value=5.0)
P_initial = st.sidebar.number_input("Initial Pollution (ppm)", value=50, min_value=0, max_value=200)
P_rate = st.sidebar.slider("Pollution Growth Rate (ppm/year)", min_value=0.0, max_value=5.0, value=2.0, step=0.5)
pollution_sd = st.sidebar.number_input("Pollution Variability (± ppm/year)", value=0.5, min_value=0.0, max_value=5.0)

# Human Impact Factors
st.sidebar.subheader("Human Impact Factors")
hunting_rate = st.sidebar.slider("Hunting Rate (% of population)", min_value=0.0, max_value=0.2, value=0.05, step=0.01)
deforestation_rate = st.sidebar.slider("Deforestation Rate (% of habitat/year)", min_value=0.0, max_value=0.05, value=0.02, step=0.01)
anti_poaching = st.sidebar.slider("Anti-Poaching Effort (% effectiveness)", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
reforestation_rate = st.sidebar.slider("Reforestation Rate (% of habitat/year)", min_value=0.0, max_value=0.05, value=0.01, step=0.01)

# Simulation Settings
st.sidebar.subheader("Simulation Settings")
time_steps = st.sidebar.number_input("Simulation Duration (Years)", value=50, min_value=10)
num_simulations = st.sidebar.number_input("Number of Monte Carlo Runs", value=10, min_value=1, max_value=1000)

# Function to calculate carrying capacity
def calculate_carrying_capacity(R, T, P, K_max):
    K_R = np.exp(-((R - 1000) / 200) ** 2)  # Effect of rainfall
    K_T = np.exp(-((T - 20) / 5) ** 2)      # Effect of temperature
    K_P = 1 - min(P / 200, 1)               # Effect of pollution
    K = K_max * K_R * K_T * K_P             # Combine the components
    return min(K, K_max)  # Clamp to ensure it does not exceed K_max

# Function to generate Monte Carlo-based temperature
def generate_temperature(T_initial, T_rate, time_steps, temp_sd):
    temperatures = []
    for t in range(time_steps):
        temperature = T_initial + T_rate * t + np.random.normal(0, temp_sd)
        temperatures.append(temperature)
    return temperatures

# Function to generate Monte Carlo-based pollution
def generate_pollution(P_initial, P_rate, time_steps, pollution_sd):
    pollution = []
    for t in range(time_steps):
        noise = np.random.normal(0, pollution_sd)
        pollution.append(P_initial + P_rate * t + noise)
    return pollution

# Simulation Function
def simulate_population(N0, r, K_max, R_mean, R_sd, T_initial, T_rate, P_initial, P_rate, pollution_sd,
                        hunting_rate, deforestation_rate, anti_poaching, reforestation_rate,
                        temp_sd, time_steps):
    N = [N0]
    K = []
    rainfall = np.random.normal(R_mean, R_sd, time_steps)
    temperature = generate_temperature(T_initial, T_rate, time_steps, temp_sd)
    pollution = generate_pollution(P_initial, P_rate, time_steps, pollution_sd)

    for t in range(time_steps):
        # Calculate carrying capacity
        K_t = calculate_carrying_capacity(rainfall[t], temperature[t], pollution[t], K_max)
        K.append(K_t)

        # Logistic growth with human impact
        dN = r * N[-1] * (1 - N[-1] / K_t) if K_t > 0 else -N[-1]
        N.append(max(N[-1] + dN, 0))  # Ensure population stays non-negative

    return N[:-1], K, rainfall, temperature, pollution

# Run Simulation
if st.sidebar.button("Run Simulation"):
    st.subheader("Simulation Results")

    all_results = []  # Store results for all simulations
    avg_population = np.zeros(time_steps)  # To calculate average population
    avg_carrying_capacity = np.zeros(time_steps)
    avg_rainfall = np.zeros(time_steps)
    avg_temperature = np.zeros(time_steps)
    avg_pollution = np.zeros(time_steps)

    best_case = None
    worst_case = None
    max_population = -1
    min_population_year = time_steps + 1

    # Run the Monte Carlo simulations
    for i in range(num_simulations):
        # Randomize parameters for Monte Carlo simulation
        random_r = np.random.uniform(r - 0.1, r + 0.1)

        # Run the simulation with randomized parameters
        N, K, rainfall, temperature, pollution = simulate_population(
            N0, random_r, K_max, R_mean, R_sd, T_initial, T_rate, P_initial, P_rate, pollution_sd,
            hunting_rate, deforestation_rate, anti_poaching, reforestation_rate, temp_sd, time_steps
        )
        all_results.append((N, K, rainfall, temperature, pollution))

        # Update averages
        avg_population += np.array(N)
        avg_carrying_capacity += np.array(K)
        avg_rainfall += np.array(rainfall)
        avg_temperature += np.array(temperature)
        avg_pollution += np.array(pollution)

        # Identify best and worst cases
        if N[-1] > max_population:
            max_population = N[-1]
            best_case = N
        extinction_year = next((year for year, pop in enumerate(N) if pop <= 0), time_steps)
        if extinction_year < min_population_year:
            min_population_year = extinction_year
            worst_case = N

    # Calculate averages
    avg_population /= num_simulations
    avg_carrying_capacity /= num_simulations
    avg_rainfall /= num_simulations
    avg_temperature /= num_simulations
    avg_pollution /= num_simulations

    # Plot 3 Cases (Best, Worst, Average)
    st.subheader("Graph: Best, Worst, and Average Cases")
    plt.figure(figsize=(10, 6))
    plt.plot(best_case, label="Best Case", color="green")
    plt.plot(worst_case, label="Worst Case", color="red")
    plt.plot(avg_population, label="Average Case", color="blue")
    plt.title("Best, Worst, and Average Cases")
    plt.xlabel("Time (Years)")
    plt.ylabel("Population Size")
    plt.legend()
    st.pyplot(plt)

    # Display Data Table for the Average Case
    st.subheader("Data for the Average Case")
    table_data_average = {
        "Year": list(range(time_steps)),
        "Average Population": avg_population,
        "Average Carrying Capacity": avg_carrying_capacity,
        "Average Rainfall (mm)": avg_rainfall,
        "Average Temperature (°C)": avg_temperature,
        "Average Pollution (ppm)": avg_pollution,
    }
    df_average = pd.DataFrame(table_data_average)
    st.dataframe(df_average)

    # Plot all Monte Carlo runs
    st.subheader("Graph: All Monte Carlo Runs")
    plt.figure(figsize=(10, 6))
    for result in all_results:
        plt.plot(result[0], alpha=0.3, color="gray")
    plt.title("All Monte Carlo Runs")
    plt.xlabel("Time (Years)")
    plt.ylabel("Population Size")
    st.pyplot(plt)
