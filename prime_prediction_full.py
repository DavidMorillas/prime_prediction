#!/usr/bin/env python
"""
Unified Experiment Script for Prime Prediction and RSA Key Generation

This script unifies all functionalities:
1. Prime generation.
2. Fractal interpolation and parameter fitting (with multiple initial guesses).
3. Error correction using ML (Random Forest and Gradient Boosting).
4. Dynamic weight optimization.
5. Extreme error correction.
6. Ultra-high range tests.
7. Execution time analysis.
8. Sensitivity analysis of hyperparameters.
9. Experimental comparison with deep learning (LSTM and Transformer).
10. Practical cryptographic application: RSA key generation using model-generated primes.
"""

import time
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sympy import isprime, nextprime, mod_inverse
from scipy.optimize import curve_fit, minimize

# For deep learning (LSTM and Transformer)
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, LayerNormalization, MultiHeadAttention
from tensorflow.keras.optimizers import Adam

# =============================================================================
# 1. Prime Generation
# =============================================================================
def generate_prime_list(start_range, end_range):
    """Generates a list of prime numbers in the range [start_range, end_range)."""
    return [n for n in range(start_range, end_range) if isprime(n)]

# =============================================================================
# 2. Fractal Interpolation and Parameter Fitting
# =============================================================================
def fractal_prime_model_safe(n, a, b, c, d):
    """Fractal interpolation model for prime prediction."""
    return a * np.log(np.abs(b * n + c) + 1) + d

def fit_fractal_prime_model_safe(start_range, end_range):
    """
    Fits the fractal model to the prime numbers obtained in the given range.
    It tries several initial guesses to avoid convergence issues.
    """
    prime_list = generate_prime_list(start_range, end_range)
    if len(prime_list) < 10:
        raise ValueError("Not enough primes in the selected range to fit the model.")
    n_indices = np.arange(1, len(prime_list) + 1)
    initial_guesses = [
        [1, 0.01, 10, 0],
        [50, 0.1, 1, 1000],
        [100, 0.05, 5, 900]
    ]
    for guess in initial_guesses:
        try:
            popt, _ = curve_fit(fractal_prime_model_safe, n_indices, prime_list, p0=guess, maxfev=20000)
            return popt, prime_list
        except RuntimeError:
            continue
    raise RuntimeError("Optimal parameters not found with any initial guess.")

def generate_fractal_primes(n_values, params):
    """Generates prime predictions using the fitted fractal model."""
    return [round(fractal_prime_model_safe(n, *params)) for n in n_values]

# =============================================================================
# 3. Prepare Data for LSTM/Transformer
# =============================================================================
def prepare_lstm_data(sequence, seq_length=10):
    """Prepares data for training an LSTM or Transformer model using sequences of length seq_length."""
    X, y = [], []
    for i in range(len(sequence) - seq_length):
        X.append(sequence[i:i+seq_length])
        y.append(sequence[i+seq_length])
    return np.array(X), np.array(y)

# =============================================================================
# 4. ML Error Correction (RF and GB)
# =============================================================================
def train_ml_error_correction(start_range, end_range, n_estimators=100):
    """
    Trains Random Forest and Gradient Boosting models to correct the residual errors
    of the fractal interpolation model.
    """
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    params, prime_list = fit_fractal_prime_model_safe(start_range, end_range)
    n_vals = np.arange(1, len(prime_list) + 1)
    fractal_preds = generate_fractal_primes(n_vals, params)
    errors = np.array(prime_list) - np.array(fractal_preds)
    X = n_vals.reshape(-1, 1)
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    gb = GradientBoostingRegressor(n_estimators=n_estimators, random_state=42)
    rf.fit(X, errors)
    gb.fit(X, errors)
    return rf, gb, params, prime_list

# =============================================================================
# 5. Dynamic Weight Optimization
# =============================================================================
def regularized_weighted_error_loss(weights, rf_preds, gb_preds, true_vals):
    """Cost function with regularization for weight optimization."""
    w1, w2 = weights
    combined = w1 * rf_preds + w2 * gb_preds
    errors = combined - true_vals
    reg = 10 * (np.abs(w1 - 0.5) + np.abs(w2 - 0.5))
    return np.mean(errors**2) + reg

def dynamic_weight_optimization(rf_preds, gb_preds, true_vals, segment_size=10000):
    """Dynamically optimizes the weights for segments of the range."""
    optimal_weights = []
    num_segments = len(true_vals) // segment_size
    for i in range(num_segments):
        start = i * segment_size
        end = start + segment_size
        seg_rf = rf_preds[start:end]
        seg_gb = gb_preds[start:end]
        seg_true = true_vals[start:end]
        res = minimize(
            regularized_weighted_error_loss,
            [0.5, 0.5],
            args=(seg_rf, seg_gb, seg_true),
            bounds=[(0,1), (0,1)],
            constraints={'type': 'eq', 'fun': lambda w: w[0] + w[1] - 1}
        )
        optimal_weights.append(res.x)
    return np.array(optimal_weights)

# =============================================================================
# 6. Extreme Error Correction
# =============================================================================
def correct_extreme_errors(predicted, true, threshold_multiplier=3):
    """Corrects extreme errors by replacing them with the mean of neighboring values."""
    errors = predicted - true
    threshold = threshold_multiplier * np.std(errors)
    corrected = predicted.copy()
    for i in range(len(errors)):
        if np.abs(errors[i]) > threshold:
            if 0 < i < len(errors) - 1:
                corrected[i] = (predicted[i-1] + predicted[i+1]) / 2
            elif i == 0:
                corrected[i] = predicted[i+1]
            else:
                corrected[i] = predicted[i-1]
    return corrected

# =============================================================================
# 7. Ultra-High Range Tests
# =============================================================================
def ultra_high_tests():
    ranges = [(10**15, 10**15+100000), (10**18, 10**18+100000)]
    results = []
    for start, end in ranges:
        try:
            primes = generate_prime_list(start, end)
            results.append({"Start Range": start, "End Range": end, "Status": "Success", "Prime Count": len(primes)})
        except Exception as e:
            results.append({"Start Range": start, "End Range": end, "Status": f"Error: {str(e)}"})
    return pd.DataFrame(results)

# =============================================================================
# 8. Execution Time Analysis
# =============================================================================
def measure_execution_time(start_range, end_range):
    start = time.time()
    generate_prime_list(start_range, end_range)
    return time.time() - start

def execution_time_analysis():
    ranges = [(10**6, 10**6+1000), (10**9, 10**9+10000), (10**12, 10**12+100000)]
    results = []
    for start, end in ranges:
        t = measure_execution_time(start, end)
        results.append({"Start Range": start, "End Range": end, "Size": end - start, "Time (s)": t})
    return pd.DataFrame(results)

# =============================================================================
# 9. Sensitivity Analysis of Hyperparameters (RF and GB)
# =============================================================================
def sensitivity_analysis_rf(param_values, start_range, end_range):
    errors = []
    primes = np.array(generate_prime_list(start_range, end_range))
    X = np.arange(1, len(primes) + 1).reshape(-1, 1)
    from sklearn.ensemble import RandomForestRegressor
    for n_est in param_values:
        rf = RandomForestRegressor(n_estimators=n_est, random_state=42)
        rf.fit(X, primes)
        preds = rf.predict(X)
        errors.append(np.mean(np.abs(preds - primes)))
    return errors

def sensitivity_analysis_gb(param_values, start_range, end_range):
    errors = []
    primes = np.array(generate_prime_list(start_range, end_range))
    X = np.arange(1, len(primes) + 1).reshape(-1, 1)
    from sklearn.ensemble import GradientBoostingRegressor
    for n_est in param_values:
        gb = GradientBoostingRegressor(n_estimators=n_est, random_state=42)
        gb.fit(X, primes)
        preds = gb.predict(X)
        errors.append(np.mean(np.abs(preds - primes)))
    return errors

def plot_sensitivity_analysis():
    param_range = np.arange(10, 110, 10)
    rf_err = sensitivity_analysis_rf(param_range, 10**3, 10**3+500)
    gb_err = sensitivity_analysis_gb(param_range, 10**3, 10**3+500)
    
    plt.figure(figsize=(8,5))
    plt.plot(param_range, rf_err, marker='o', label="Random Forest")
    plt.plot(param_range, gb_err, marker='s', label="Gradient Boosting")
    plt.xlabel("Number of Trees")
    plt.ylabel("Mean Absolute Error")
    plt.title("Hyperparameter Sensitivity Analysis")
    plt.legend()
    plt.grid(True)
    plt.show()

# =============================================================================
# 10. Experimental Comparison with Deep Learning (LSTM)
# =============================================================================
def train_lstm_model(start_range, end_range, seq_length=10, epochs=30, batch_size=32):
    params, primes = fit_fractal_prime_model_safe(start_range, end_range)
    n_vals = np.arange(1, len(primes) + 1)
    fractal_preds = generate_fractal_primes(n_vals, params)
    errors = np.array(primes) - np.array(fractal_preds)
    X, y = prepare_lstm_data(errors, seq_length)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(0.2),
        LSTM(100, return_sequences=False),
        Dropout(0.2),
        Dense(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)
    return model, primes, params

def evaluate_lstm_model(model, primes, params, seq_length=10):
    n_vals = np.arange(1, len(primes) + 1)
    fractal_preds = generate_fractal_primes(n_vals, params)
    errors = np.array(primes) - np.array(fractal_preds)
    X, _ = prepare_lstm_data(errors, seq_length)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    lstm_preds = model.predict(X, verbose=0).flatten()
    corrected_preds = np.array(fractal_preds)[seq_length:] + lstm_preds
    true_vals = np.array(primes)[seq_length:]
    error_vals = corrected_preds - true_vals
    return error_vals

def plot_lstm_results():
    start_range = 10**6
    end_range = start_range + 10000
    model, primes, params = train_lstm_model(start_range, end_range, epochs=30, batch_size=32)
    error_vals = evaluate_lstm_model(model, primes, params, seq_length=10)
    
    print("LSTM Model Error Statistics:")
    print(f"Mean Error: {np.mean(error_vals):.4f}")
    print(f"Median Error: {np.median(error_vals):.4f}")
    print(f"Max Error: {np.max(error_vals):.4f}")
    print(f"Min Error: {np.min(error_vals):.4f}")
    print(f"Std Dev: {np.std(error_vals):.4f}")
    
    plt.figure(figsize=(8,5))
    plt.hist(error_vals, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel("Error (Prediction - Actual)")
    plt.ylabel("Frequency")
    plt.title("LSTM Model Error Histogram")
    plt.grid(True)
    plt.show()

# =============================================================================
# 11. Experimental Comparison with Advanced Deep Learning Models (Transformer)
# =============================================================================
def transformer_block(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    """Defines a single Transformer block."""
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(x + inputs)
    x_ff = Dense(ff_dim, activation="relu")(x)
    x_ff = Dense(inputs.shape[-1])(x_ff)
    x_ff = Dropout(dropout)(x_ff)
    return LayerNormalization(epsilon=1e-6)(x + x_ff)

def train_transformer_model(start_range, end_range, seq_length=10, epochs=30, batch_size=32):
    """
    Trains a Transformer-based model to correct the residual errors from the fractal model.
    This function reuses the same data preparation as the LSTM experiment.
    """
    params, primes = fit_fractal_prime_model_safe(start_range, end_range)
    n_vals = np.arange(1, len(primes) + 1)
    fractal_preds = generate_fractal_primes(n_vals, params)
    errors = np.array(primes) - np.array(fractal_preds)
    X, y = prepare_lstm_data(errors, seq_length)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    inputs = Input(shape=(seq_length, 1))
    x = transformer_block(inputs, head_size=32, num_heads=2, ff_dim=64, dropout=0.1)
    x = transformer_block(x, head_size=32, num_heads=2, ff_dim=64, dropout=0.1)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.1)(x)
    # Use the output from the last time step for prediction
    x = Dense(1)(x[:, -1, :])
    model = Model(inputs, x)
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)
    return model, primes, params

def evaluate_transformer_model(model, primes, params, seq_length=10):
    """
    Evaluates the trained Transformer model by computing the residual errors after
    applying the correction to the fractal model predictions.
    """
    n_vals = np.arange(1, len(primes) + 1)
    fractal_preds = generate_fractal_primes(n_vals, params)
    errors = np.array(primes) - np.array(fractal_preds)
    X, _ = prepare_lstm_data(errors, seq_length)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    transformer_preds = model.predict(X, verbose=0).flatten()
    corrected_preds = np.array(fractal_preds)[seq_length:] + transformer_preds
    true_vals = np.array(primes)[seq_length:]
    error_vals = corrected_preds - true_vals
    return error_vals

def plot_transformer_results():
    """
    Trains and evaluates the Transformer model, then plots the error histogram and
    prints key statistics.
    """
    start_range = 10**6
    end_range = start_range + 10000
    model, primes, params = train_transformer_model(start_range, end_range, epochs=30, batch_size=32)
    error_vals = evaluate_transformer_model(model, primes, params, seq_length=10)
    
    print("Transformer Model Error Statistics:")
    print(f"Mean Error: {np.mean(error_vals):.4f}")
    print(f"Median Error: {np.median(error_vals):.4f}")
    print(f"Max Error: {np.max(error_vals):.4f}")
    print(f"Min Error: {np.min(error_vals):.4f}")
    print(f"Std Dev: {np.std(error_vals):.4f}")
    
    plt.figure(figsize=(8,5))
    plt.hist(error_vals, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel("Error (Prediction - Actual)")
    plt.ylabel("Frequency")
    plt.title("Transformer Model Error Histogram")
    plt.grid(True)
    plt.show()

# =============================================================================
# 12. Practical Cryptographic Application: RSA Key Generation
# =============================================================================
def generate_model_based_prime(index, params):
    """
    Generates a candidate prime using the fractal model.
    If the rounded candidate is not prime, the next prime is chosen.
    """
    candidate = round(fractal_prime_model_safe(index, *params))
    if not isprime(candidate):
        candidate = nextprime(candidate)
    return candidate

def generate_rsa_key_from_model(params):
    """
    Generates an RSA key using two primes obtained from the fractal model.
    Random indices (within a predefined interval) are chosen to extrapolate large primes.
    """
    p_index = random.randint(10**5, 10**6)
    q_index = random.randint(10**5, 10**6)
    
    p_candidate = generate_model_based_prime(p_index, params)
    q_candidate = generate_model_based_prime(q_index, params)
    
    while p_candidate == q_candidate:
        q_index = random.randint(10**5, 10**6)
        q_candidate = generate_model_based_prime(q_index, params)
    
    n = p_candidate * q_candidate
    phi = (p_candidate - 1) * (q_candidate - 1)
    e = 65537
    d = mod_inverse(e, phi)
    
    return {"p": p_candidate, "q": q_candidate, "n": n, "e": e, "d": d}

def run_rsa_application():
    """
    Fits the fractal model over a known range and uses it to generate RSA key components.
    """
    try:
        params, _ = fit_fractal_prime_model_safe(1000, 1100)
    except RuntimeError as e:
        print("Error fitting fractal model for RSA application:", e)
        return
    rsa_keys = generate_rsa_key_from_model(params)
    print("\n=== RSA Key Generation Application ===")
    print("p =", rsa_keys["p"])
    print("q =", rsa_keys["q"])
    print("n =", rsa_keys["n"])
    print("e =", rsa_keys["e"])
    print("d =", rsa_keys["d"])

# =============================================================================
# 13. Main: Execute All Experiments and Applications
# =============================================================================
def main():
    print("=== Unified Experiments for Prime Prediction ===\n")
    
    # 1. Prime generation test
    print("Generating primes in a small range:")
    primes_sample = generate_prime_list(1000, 1100)
    print(primes_sample, "\n")
    
    # 2. Fractal model fitting
    print("Fitting fractal model:")
    try:
        params, primes = fit_fractal_prime_model_safe(1000, 1100)
        print("Fitted parameters:", params, "\n")
    except RuntimeError as e:
        print("Error in fitting fractal model:", e)
        return
    
    # 3. Train ML error correction (RF and GB)
    print("Training RF and GB models for error correction:")
    rf_model, gb_model, params_ml, primes_ml = train_ml_error_correction(1000, 1200, n_estimators=50)
    print("ML models trained.\n")
    
    # 4. Dynamic weight optimization (synthetic example)
    print("Performing dynamic weight optimization (synthetic example):")
    synthetic_rf = np.random.rand(100)
    synthetic_gb = np.random.rand(100)
    synthetic_true = 0.5 * synthetic_rf + 0.5 * synthetic_gb + np.random.normal(0, 0.1, 100)
    opt_weights = dynamic_weight_optimization(synthetic_rf, synthetic_gb, synthetic_true, segment_size=20)
    print("Optimal weights per segment:\n", opt_weights, "\n")
    
    # 5. Ultra-high range tests
    print("Running ultra-high range tests:")
    ultra_df = ultra_high_tests()
    print(ultra_df, "\n")
    
    # 6. Execution time analysis
    print("Execution time analysis:")
    exec_df = execution_time_analysis()
    print(exec_df, "\n")
    plt.figure(figsize=(8,5))
    plt.plot(exec_df["Size"], exec_df["Time (s)"], marker='o')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Range Size")
    plt.ylabel("Time (s)")
    plt.title("Execution Time Analysis")
    plt.grid(True, which="both", linestyle="--")
    plt.show()
    
    # 7. Sensitivity analysis
    print("Running sensitivity analysis:")
    plot_sensitivity_analysis()
    
    # 8. Deep learning (LSTM) experimental comparison
    print("Training and evaluating LSTM model (Deep Learning Comparison):")
    plot_lstm_results()
    
    # 9. Advanced Deep Learning Comparison: Transformer Model
    print("Training and evaluating Transformer model (Advanced Deep Learning Comparison):")
    plot_transformer_results()
    
    # 10. Practical Cryptographic Application: RSA Key Generation
    run_rsa_application()

if __name__ == "__main__":
    main()
