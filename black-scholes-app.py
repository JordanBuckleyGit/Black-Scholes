import streamlit as st
import numpy as np
from scipy.stats import norm # for the cumulative distribution function (CDF)
import matplotlib.pyplot as plt
import pandas as pd # for easier data handling for charts

# --- Core Black-Scholes and Greeks Algorithms ---

def normal_cdf(x):
    # calculates cumulative distribution func of the standard normal distribution
    return norm.cdf(x)

def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    # calculates the black scholes option price. will implement the formula here and handle edge cases
    return price

def calculate_delta(S, K, T, r, sigma, option_type='call'):
    # calculates delta of an option. will implement the formula here

def calculate_gamma(S, K, T, r, sigma):
    # calculates the gamma of an option. will implement the formula here

def calculate_vega(S, K, T, r, sigma):
    # calculates the vega of an option. will implement the formula here

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
    
# streamlit app layout to be implemented below