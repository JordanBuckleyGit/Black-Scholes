import streamlit as st
import numpy as np
from scipy.stats import norm # for the cumulative distribution function (CDF)
import matplotlib.pyplot as plt
import pandas as pd # for easier data handling for charts
import seaborn as sns

# --- Core Black-Scholes and Greeks Algorithms ---
st.set_page_config(
    page_title="Black-Scholes Option Pricing Model",
    page_icon="📊",
    initial_sidebar_state="expanded",
    layout="wide")

def normal_cdf(x):
    return norm.cdf(x) # calculates cumulative distribution func of the standard normal distribution

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
    return theta / 365 # return per day theta

def calculate_rho(S, K, T, r, sigma, option_type='call'):
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        rho = K * T * np.exp(-r * T) * normal_cdf(d2)
    elif option_type == 'put':
        rho = -K * T * np.exp(-r * T) * normal_cdf(-d2)
    else:
         raise ValueError("Option Type must be 'call' or 'put'")

    return rho / 100 # returns as a percentage

###################################
# Box
###################################

def display_input_summary(S, K, T, sigma, r):
    input_df = pd.DataFrame({
        "Current Asset Price": [S],
        "Strike Price": [K],
        "Time to Maturity (Years)": [T],
        "Volatility (σ)": [sigma],
        "Risk-Free Interest Rate": [r]
    })
    st.dataframe(input_df.style.format("{:.4f}"), use_container_width=True)

def display_option_value_cards(S, K, T, r, sigma):
    call_value = black_scholes_price(S, K, T, r, sigma, "call")
    put_value = black_scholes_price(S, K, T, r, sigma, "put")
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown(
            f"""
            <div style="background-color:#7fdc8c;padding:10px 0 10px 0;border-radius:15px;text-align:center;">
                <span style="font-size:16px;font-weight:bold;color:black;">CALL Value</span><br>
                <span style="font-size:22px;font-weight:bold;color:black;">${call_value:.2f}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"""
            <div style="background-color:#ffb3b3;padding:10px 0 10px 0;border-radius:15px;text-align:center;">
                <span style="font-size:16px;font-weight:bold;color:black;">PUT Value</span><br>
                <span style="font-size:22px;font-weight:bold;color:black;">${put_value:.2f}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

def plot_bs_heatmaps(spot_range, vol_range, K, T, r):
    call_prices = np.zeros((len(vol_range), len(spot_range)))
    put_prices = np.zeros((len(vol_range), len(spot_range)))
    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            call_prices[i, j] = black_scholes_price(spot, K, T, r, vol, "call")
            put_prices[i, j] = black_scholes_price(spot, K, T, r, vol, "put")
    cmap = sns.diverging_palette(10, 130, as_cmap=True)
    fig_call, ax_call = plt.subplots(figsize=(8, 6))
    sns.heatmap(call_prices, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2),
                annot=True, fmt=".2f", cmap=cmap, ax=ax_call, cbar_kws={'label': 'Call Price'})
    ax_call.set_title('CALL Option Price Heatmap')
    ax_call.set_xlabel('Spot Price')
    ax_call.set_ylabel('Volatility')
    fig_put, ax_put = plt.subplots(figsize=(8, 6))
    sns.heatmap(put_prices, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2),
                annot=True, fmt=".2f", cmap=cmap, ax=ax_put, cbar_kws={'label': 'Put Price'})
    ax_put.set_title('PUT Option Price Heatmap')
    ax_put.set_xlabel('Spot Price')
    ax_put.set_ylabel('Volatility')
    return fig_call, fig_put

###############################
# UI (sidebar)
##############################
st.title("📊 Black-Scholes Model")

with st.sidebar: # used for simplicity 
    st.markdown("""
    ### Created by: Jordan Buckley
    """)
    st.markdown("""
    <a href="https://www.linkedin.com/in/jordan05/" target="_blank" style="text-decoration: none;">
        <img src="https://img.icons8.com/ios-filled/24/000000/linkedin.png"/> LinkedIn
    </a>
    <a href="https://github.com/JordanBuckleyGit" target="_blank" style="text-decoration: none;">
        <img src="https://img.icons8.com/ios-filled/24/000000/github.png"/> GitHub
    </a>
    <a href="mailto:jordanbuckleycork@gmail.com" target="_blank" style="text-decoration: none;">
        <img src="https://img.icons8.com/ios-filled/24/000000/new-post.png"/> Contact
    </a>                   
    """, unsafe_allow_html=True)
    st.markdown("---")
    S = st.number_input(
        "Spot Price (S)", 
        min_value=1.0, 
        max_value=500.0, 
        value=100.0, 
        step=1.0,
        help="The current price of the underlying asset."
    )
    K = st.number_input(
        "Strike price (K)", 
        min_value=1.0, 
        max_value=500.0, 
        value=100.0, 
        step=1.0,
        help="The price at which you have the right to buy (call) or sell (put) the asset."
    )
    T = st.number_input(
        "Time to Maturity (Years, T)", 
        min_value=0.01, 
        max_value=5.0, 
        value=1.0, 
        step=0.01,
        help="The time in years until the option expires."
    )
    r = st.number_input(
        "Risk-Free Rate (r, decimal)", 
        min_value=0.0, 
        max_value=0.2, 
        value=0.05, 
        step=0.001,
        help="The annualized risk-free interest rate (as a decimal, e.g., 0.05 for 5%)."
    )
    sigma = st.number_input(
        "Volatility (σ, decimal)", 
        min_value=0.01, 
        max_value=1.0, 
        value=0.2, 
        step=0.01,
        help="The annualized volatility of the underlying asset (as a decimal, e.g., 0.2 for 20%)."
    )
    option_type = st.selectbox(
        "Option Type", 
        ["call", "put"],
        help="Choose 'call' for the right to buy, or 'put' for the right to sell."
    )
    st.markdown("---")
    st.subheader("PNL")
    option_price = st.number_input(
        "Option Price (Premium)", min_value=0.0, value=float(f"{black_scholes_price(S, K, T, r, sigma, option_type):.2f}"), step=0.01
    )
    num_contracts = st.number_input("Number of Option Contracts", min_value=1, value=1, step=1)
    calculate_button = st.button("Calculate")
    st.markdown("---")
    calculate_btn = st.button("Heatmap Parameters")
    spot_min = st.number_input("Min Spot Price", min_value=0.01, value=S*0.8, step=0.01)
    spot_max = st.number_input("Max Spot Price", min_value=0.01, value=S*1.2, step=0.01)
    vol_min = st.slider("Min Volatility for Heatmap", min_value=0.01, max_value=1.0, value=sigma*0.5, step=0.01)
    vol_max = st.slider("Max Volatility for Heatmap", min_value=0.01, max_value=1.0, value=sigma*1.5, step=0.01)
    
    spot_range = np.linspace(spot_min, spot_max, 10)
    vol_range = np.linspace(vol_min, vol_max, 10)

display_input_summary(S, K, T, sigma, r)
display_option_value_cards(S, K, T, r, sigma)

# Graphing stuff

# if calculate_button:
# heatmap graph
st.header("Options Price - Heatmaps")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Call Price Heatmap")
    fig_call, fig_put = plot_bs_heatmaps(spot_range, vol_range, K, T, r)
    st.pyplot(fig_call)
with col2:
    st.subheader("Put Price Heatmap")
    st.pyplot(fig_put)

# --- Option Payoff at Expiry (P&L) ---
st.header("Option Payoff at Expiry (PNL)")
S_range = np.linspace(max(1.0, S - 50), S + 50, 100)
contract_size = 100  # standard contract size for options

if option_type == "call":
    payoff = np.maximum(S_range - K, 0) - option_price
else:
    payoff = np.maximum(K - S_range, 0) - option_price

total_payoff = payoff * num_contracts * contract_size

fig_payoff, ax_payoff = plt.subplots(figsize=(10, 6))
ax_payoff.plot(S_range, total_payoff, label=f'{option_type.capitalize()} Option P&L', color='royalblue')
ax_payoff.axhline(0, color='black', linestyle='--', linewidth=1)
ax_payoff.axvline(K, color='red', linestyle=':', linewidth=1, label='Strike Price')
ax_payoff.set_title(f"{option_type.capitalize()} Option Payoff at Expiry")
ax_payoff.set_xlabel("Spot Price at Expiry (S)")
ax_payoff.set_ylabel("Profit / Loss (€)")
ax_payoff.grid(True, linestyle='--', alpha=0.7)
ax_payoff.legend()
st.pyplot(fig_payoff)

contract_size = 100  # standard contract size for options
total_cost = option_price * num_contracts * contract_size
intrinsic_value = max(0, S - K) if option_type == "call" else max(0, K - S)
potential_profit = (intrinsic_value - option_price) * num_contracts * contract_size
potential_return = (potential_profit / total_cost * 100) if total_cost > 0 else 0

st.subheader("Profit & Loss (P&L)")
col1, col2, col3 = st.columns(3)
col1.metric("Total Option Cost (Premium)", f"€ {total_cost:,.2f}")
col2.metric("Potential Profit", f"€ {potential_profit:,.2f}")
col3.metric("Potential Return", f"{potential_return:.2f} %")

if potential_profit > 0:
    st.success("Profitable! 🎉")
else:
    st.info("Unprofitable")
