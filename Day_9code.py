import numpy as np
import matplotlib.pyplot as plt

""" 
 Logistic Regression equation

"""
# Example coefficients
b0 = -10
b1 = 0.3

# Ages
ages = np.linspace(10, 60, 200)

# Linear part
z = b0 + b1 * ages

# Sigmoid function
probability = 1 / (1 + np.exp(-z))


""" 
 Display some calculations
"""

for age in [10, 20, 30, 40, 50, 60]:

    z_value = b0 + b1 * age

    p = 1 / (1 + np.exp(-z_value))

    print(
        f"Age: {age} | "
        f"z = {z_value:.2f} | "
        f"Probability = {p:.2%}"
    )


""" 
Draw the sigmoid curve
"""

plt.plot(ages, probability)

plt.axhline(
    y=0.5,
    linestyle="--",
    label="Threshold = 0.5"
)

plt.xlabel("Age")
plt.ylabel("Probability of buying")
plt.title("Logistic Regression - Sigmoid Function")

plt.legend()
plt.grid()

plt.show()