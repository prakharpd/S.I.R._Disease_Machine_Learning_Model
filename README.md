31/5/26: There is small bug in Github in which Jupyternotbook previews are not visible. So in many of my repos you have to download it to look that file locally.


# 🦠 S.I.R. Disease Machine Learning Model: From Linear Regression to Deep Learning

---

## 📑 Table of Contents
* [1. The Problem Statement](#1-the-problem-statement)
* [2. Version 1: The Linear Regression Approach](#2-version-1-the-linear-regression-approach)
  * [The Implementation](#the-v1-implementation)
  * [The Mathematical Trap](#the-mathematical-trap-why-v1-failed)
* [3. Version 2: The Hybrid ANN Solution](#3-version-2-the-hybrid-ann-solution)
  * [Residual Error Correction](#how-it-works-residual-error-correction)
  * [Final Performance Metrics](#v2-final-results)
* [4. Tech Stack](#4-tech-stack)

---

## 1. The Problem Statement

Predicting the spread of infectious diseases is a highly complex challenge. Traditional epidemiological models, such as the **SIR (Susceptible-Infected-Recovered) Model**, rely on rigid differential equations governed by two critical parameters:
* **$\beta$ (Beta):** The transmission rate.
* **$\gamma$ (Gamma):** The recovery rate.

**The Goal:** Accurately determine these parameters from historical data to build a predictive model, and engineer a system that can adapt to the non-linear, unpredictable fluctuations inherent in real-world human behavior.

---

## 2. Version 1: The Linear Regression Approach

In my first iteration, I attempted to approach the parameter discovery problem using standard Machine Learning techniques via Scikit-Learn's `LinearRegression` model.

<details>
<summary><b>🛠️ Click to expand: The V1 Implementation Details</b></summary>

1. **Data Ingestion:** Loaded the historical `Disease.csv` dataset, extracting the features ($x$) and targets ($y$).
2. **Training & Validation:** Split the data using an 80/20 Train/Test ratio (`train_test_split`).
3. **Parameter Extraction:** I attempted to extract the mathematical SIR parameters directly from the trained linear model's attributes:
   * $\beta$ was derived from the slope coefficient (`regressor.coef_[0]`).
   * $\gamma$ was derived from the y-intercept (`regressor.intercept_`).
4. **ODE Simulation:** Passed user-defined initial conditions ($S_0=100, I_0=1, R_0=0$) and the extracted parameters into SciPy's `odeint` integrator.
</details>

### The Mathematical Trap (Why V1 Failed)
While the Linear Regression achieved a seemingly high $R^2$ of **0.91** on its specific testing slice, with an assumption that data set variation would be linear, each time. 

In a biological SIR model, these rates typically sit between $0.0$ and $1.0$. This revealed two fatal flaw in our assumption in using Linear Regression for this task:
1. **The Negative Population Trap:** A straight regression line extends infinitely. When projected, it can output negative values for a total population, which is biologically impossible.
2. **The Noise Trap:** Linear regression expects data to be clean and have a linear relationship in ideal situations. In reality, epidemiological data is exponential, curved, and highly noisy. A straight line cannot capture the dynamic relationships required for differential equations, resulting in a constant, flat-line simulation graph.

---

## 3. Version 2: The Hybrid ANN Solution

To solve the limitations of V1, I engineered an advanced hybrid architecture. Instead of using Machine Learning to *guess* the base parameters, I used Deep Learning to *correct* a properly optimized mathematical baseline.

<details>
<summary><b>🧠 Click to expand: How Residual Error Correction Works</b></summary>

1. **Proper Optimization:** Used SciPy's `minimize` (Nelder-Mead algorithm) to correctly solve for $\beta$ and $\gamma$, keeping them within realistic biological bounds to generate a perfect theoretical baseline.
2. **The Target:** Calculated the residual error for every time step: 
   $$Error = Data_{true} - Model_{SIR}$$
3. **The Deep Learning Engine:** Trained a multi-layer Dense Artificial Neural Network (ANN) using Keras. The ANN takes the timeline and theoretical states as inputs to predict the expected real-world error for any given day.
4. **The Synthesis:** Added the ANN's predicted correction back into the rigid SIR baseline.
   $$Prediction_{hybrid} = Model_{SIR} + ANN_{correction}$$
</details>

### V2 Final Results
This architecture ensures the model respects the biological limits of population growth while perfectly catching real-world data spikes (like weekend reporting lags or superspreader events).

* **R² (Goodness of Fit):** **0.8870**
* **RMSE (Root Mean Squared Error):** **19.3549**
* **MAE (Mean Absolute Error):** **15.2567**

---

## 4. Tech Stack
* **Mathematics:** `SciPy` (ODE integration, Nelder-Mead Optimization)
* **Data Processing:** `NumPy`, `Pandas`, `Scikit-Learn` (Linear Regression, StandardScaler)
* **Deep Learning:** `TensorFlow` / `Keras` (Dense Sequential API)
* **Visualization:** `Matplotlib`
