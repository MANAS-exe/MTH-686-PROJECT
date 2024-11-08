import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares
from scipy.stats import shapiro

# Load your data (replace 'data.csv' with your actual dataset path)
input_file = 'data-19.txt'
data = pd.read_csv(input_file, delim_whitespace=True, header=None, names=['x', 'y'])  # Replace with your actual data file
t = data['x'].values
y = data['y'].values

# Model Definitions
def model1(t, alpha0, alpha1, beta1, alpha2, beta2):
    return alpha0 + alpha1 * np.exp(beta1 * t) + alpha2 * np.exp(beta2 * t)

def model2(t, alpha0, alpha1, beta0, beta1):
    return (alpha0 + alpha1 * t) / (beta0 + beta1 * t)

def model3(t, beta0, beta1, beta2, beta3, beta4):
    return beta0 + beta1 * t + beta2 * t**2 + beta3 * t**3 + beta4 * t**4

# Dictionary to map model names to functions
models = {
    "Model 1": model1,
    "Model 2": model2,
    "Model 3": model3
}

# Step 1: Fit Models to Data and Track Loss
initial_guess_model1 = [1, 1, -1, 1, -1]
initial_guess_model2 = [1, 1, 1, -1]
initial_guess_model3 = [1, 1, 1, 1, 1]

# Store losses
losses = {
    "Model 1": [],
    "Model 2": [],
    "Model 3": []
}

# Fit each model and store SSR (Sum of Squared Residuals)
try:
    params_model1, _ = curve_fit(model1, t, y, p0=initial_guess_model1, maxfev=5000)
    y_pred1 = model1(t, *params_model1)
    residuals1 = y - y_pred1
    losses["Model 1"].append(np.sum(residuals1 ** 2))
except RuntimeError:
    print("Model 1 did not converge")

try:
    params_model2, _ = curve_fit(model2, t, y, p0=initial_guess_model2, maxfev=5000)
    y_pred2 = model2(t, *params_model2)
    residuals2 = y - y_pred2
    losses["Model 2"].append(np.sum(residuals2 ** 2))
except RuntimeError:
    print("Model 2 did not converge")

try:
    params_model3, _ = curve_fit(model3, t, y, p0=initial_guess_model3, maxfev=5000)
    y_pred3 = model3(t, *params_model3)
    residuals3 = y - y_pred3
    losses["Model 3"].append(np.sum(residuals3 ** 2))
except RuntimeError:
    print("Model 3 did not converge")

# Step 2: Plot Loss Function
plt.figure(figsize=(8, 6))
for model_name, loss_list in losses.items():
    plt.plot(loss_list, label=f"{model_name} Loss (SSR)", marker='o')
plt.xlabel('Iteration')
plt.ylabel('Sum of Squared Residuals (SSR)')
plt.title('Loss Function Progression')
plt.legend()
plt.grid(True)
plt.show()

# Step 3: Determine Best Model Based on SSR
best_model = min(losses, key=lambda k: losses[k][-1])
best_fit_params = params_model1 if best_model == "Model 1" else (params_model2 if best_model == "Model 2" else params_model3)
y_best_fit = y_pred1 if best_model == "Model 1" else (y_pred2 if best_model == "Model 2" else y_pred3)

print(f"Best-fit Model: {best_model}")
print("Parameters:", best_fit_params)


sigma_squared = np.var(y - y_best_fit)
print(f"Estimated σ^2: {sigma_squared}")


try:
    _, covariance = curve_fit(models[best_model], t, y, p0=best_fit_params, maxfev=5000)
    parameter_variances = np.diag(covariance)
    confidence_intervals = [
        (param - 1.96 * np.sqrt(var), param + 1.96 * np.sqrt(var))
        for param, var in zip(best_fit_params, parameter_variances)
    ]
    print("Confidence Intervals for parameters:", confidence_intervals)
except RuntimeError:
    print("Could not calculate confidence intervals for the best-fit model due to convergence issues.")


plt.figure(figsize=(8, 6))
plt.scatter(t, y - y_best_fit, label='Residuals', color='purple')
plt.axhline(0, color='black', linestyle='--')
plt.xlabel('t')
plt.ylabel('Residuals')
plt.title(f'Residuals for {best_model}')
plt.legend()
plt.grid(True)
plt.show()

stat, p_value = shapiro(y - y_best_fit)
print(f"Shapiro-Wilk Test for Normality: Statistic={stat}, p-value={p_value}")
if p_value > 0.05:
    print("Residuals likely follow a normal distribution.")
else:
    print("Residuals likely do not follow a normal distribution.")

plt.figure(figsize=(8, 6))
plt.scatter(t, y, label='Observed Data', color='blue')
plt.plot(t, y_best_fit, label=f'Fitted Curve ({best_model})', color='red')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title(f'Observed Data and Fitted Curve for {best_model}')
plt.legend()
plt.grid(True)
plt.show()
