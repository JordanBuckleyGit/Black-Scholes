import streamlit as st
import numpy as np
from scipy.stats import norm # for the cumulative distribution function (CDF)
import matplotlib.pyplot as plt
import pandas as pd # for easier data handling for charts
import seaborn as sns

# --- Core Black-Scholes and Greeks Algorithms ---

def normal_cdf(x):
    # calculates cumulative distribution func of the standard normal distribution
    return norm.cdf(x)

def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Option Type must be 'call' or 'put'")
    return price # returns the black-scholes price for the option

def calculate_delta(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == "call":
        delta = norm.cdf(d1)
    elif option_type == "put":
        delta = norm.cdf(d1) - 1
    else:
         raise ValueError("Option Type must be 'call' or 'put'")
    return delta # returns the option delta (sensitivity to underlying price)

def calculate_gamma(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma # returns the rate of change of delta with respect to the underlying asset price (gamma)

def calculate_vega(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    pdf_d1 = norm.pdf(d1)
    vega = S * pdf_d1 * np.sqrt(T)
    return vega / 100 # return vega per 1% change

def calculate_theta(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    if option_type == "call":
        theta = (-S * pdf_d1 * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2))
    elif option_type == "put":
        theta = (-S * pdf_d1 * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2))
    else:
        raise ValueError("Option Type must be 'call' or 'put'")
    # return per day theta
    return theta / 365

def calculate_rho(S, K, T, r, sigma, option_type='call'):
    # calculates d2
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        rho = K * T * np.exp(-r * T) * normal_cdf(d2)
    elif option_type == 'put':
        rho = -K * T * np.exp(-r * T) * normal_cdf(-d2)
    else:
         raise ValueError("Option Type must be 'call' or 'put'")

    return rho / 100 # returns as a percentage
    
st.title("Black-Scholes Option Pricing Calculator")

st.sidebar.header("Input Parameters")
S = st.sidebar.number_input("Spot Price (S)", min_value=1.0, max_value=500.0, value=100.0, step=1.0)
K = st.sidebar.number_input("Strike price (K)", min_value=1.0, max_value=500.0, value=100.0, step=1.0)
T = st.sidebar.number_input("Time to Maturity (Years, T)", min_value=0.01, max_value=5.0, value=1.0, step=0.01)
r = st.sidebar.number_input("Risk-Free Rate (r, decimal)", min_value=0.0, max_value=0.2, value=0.05, step=0.001)
sigma = st.sidebar.number_input("Volatility (σ, decimal)", min_value=0.01, max_value=1.0, value=0.2, step=0.01)
option_type = st.sidebar.selectbox("Option Type", ["call", "put"])
calculate_button = st.sidebar.button("Calculate")

if calculate_button:
    st.header("Option Price vs. Spot Price")
    S_range = np.linspace(max(1.0, S - 50), S + 50, 100)
    prices = [black_scholes_price(s, K, T, r, sigma, option_type) for s in S_range]

    fig_line, ax_line = plt.subplots(figsize=(10, 6))
    ax_line.plot(S_range, prices, label=f'{option_type.capitalize()} Option Price')
    ax_line.set_title(f"{option_type.capitalize()} Option Price vs. Spot Price")
    ax_line.set_xlabel("Spot Price (S)")
    ax_line.set_ylabel("Option Price")
    ax_line.grid(True)
    ax_line.legend()
    st.pyplot(fig_line)

    st.header("Option Price Heatmap: Spot Price vs. Volatility")
    S_heatmap_range = np.linspace(max(1.0, S - 50), S + 50, 20)
    sigma_heatmap_range = np.linspace(0.05, 0.5, 10)

    S_grid, sigma_grid = np.meshgrid(S_heatmap_range, sigma_heatmap_range)
    price_matrix = np.array([black_scholes_price(s, K, T, r, sig, option_type)
                             for s, sig in zip(S_grid.flatten(), sigma_grid.flatten())]).reshape(sigma_grid.shape)

    fig_heatmap, ax_heatmap = plt.subplots(figsize=(12, 8))
    cmap = sns.diverging_palette(10, 130, as_cmap=True)
    sns.heatmap(price_matrix,
                cmap=cmap,
                annot=True,
                fmt=".1f",
                xticklabels=[f"{val:.0f}" for val in S_heatmap_range],
                yticklabels=[f"{val:.2f}" for val in sigma_heatmap_range],
                cbar_kws={'label': 'Option Price'},
                ax=ax_heatmap)
    ax_heatmap.set_title(f"{option_type.capitalize()} Option Price Heatmap (Red-Green Scale)")
    ax_heatmap.set_xlabel("Spot Price (S)")
    ax_heatmap.set_ylabel("Volatility (σ)")
    plt.yticks(rotation=0)
    st.pyplot(fig_heatmap)

    st.subheader("Black-Scholes Calculated Values")
    st.write(f"**Option Price:** {black_scholes_price(S, K, T, r, sigma, option_type):.4f}")
    st.write(f"**Delta:** {calculate_delta(S, K, T, r, sigma, option_type):.4f}")
    st.write(f"**Gamma:** {calculate_gamma(S, K, T, r, sigma):.4f}")
    st.write(f"**Vega:** {calculate_vega(S, K, T, r, sigma):.4f}")
    st.write(f"**Theta (per day):** {calculate_theta(S, K, T, r, sigma, option_type):.4f}")
    st.write(f"**Rho (per 1% change):** {calculate_rho(S, K, T, r, sigma, option_type):.4f}")