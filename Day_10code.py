from sklearn.svm import SVC
import matplotlib.pyplot as plt

# -----------------------------------------
# 1. Training data
# -----------------------------------------
# [weight in grams, sweetness from 0 to 10]

X = [
    [150, 8],  # Banana
    [160, 9],  # Banana
    [140, 7],  # Banana
    [170, 8],  # Banana

    [180, 3],  # Apple
    [190, 4],  # Apple
    [200, 2],  # Apple
    [175, 4],  # Apple
]

# 0 = Banana
# 1 = Apple
y = [0, 0, 0, 0, 1, 1, 1, 1]


# -----------------------------------------
# 2. Create the SVM model
# -----------------------------------------

model = SVC(kernel="linear")

# Train the model
model.fit(X, y)


# -----------------------------------------
# 3. Test new fruits
# -----------------------------------------

new_fruits = [
    [155, 8],   # New fruit
    [195, 3],   # New fruit
    [165, 7],   # New fruit
]

predictions = model.predict(new_fruits)


# -----------------------------------------
# 4. Show predictions
# -----------------------------------------

for fruit, prediction in zip(new_fruits, predictions):

    if prediction == 0:
        print(f"{fruit} -> Banana 🍌")
    else:
        print(f"{fruit} -> Apple 🍎")


# -----------------------------------------
# 5. Visualize the data
# -----------------------------------------

bananas = [X[i] for i in range(len(X)) if y[i] == 0]
apples = [X[i] for i in range(len(X)) if y[i] == 1]

plt.scatter(
    [fruit[0] for fruit in bananas],
    [fruit[1] for fruit in bananas],
    label="Bananas"
)

plt.scatter(
    [fruit[0] for fruit in apples],
    [fruit[1] for fruit in apples],
    label="Apples"
)

# Get the SVM decision boundary
w = model.coef_[0]
b = model.intercept_[0]

x1 = 130
x2 = 210

y1 = -(w[0] * x1 + b) / w[1]
y2 = -(w[0] * x2 + b) / w[1]

plt.plot(
    [x1, x2],
    [y1, y2],
    label="Decision Boundary"
)

# Show the Support Vectors
support_vectors = model.support_vectors_

plt.scatter(
    support_vectors[:, 0],
    support_vectors[:, 1],
    s=120,
    facecolors="none",
    edgecolors="black",
    label="Support Vectors"
)

plt.xlabel("Weight (grams)")
plt.ylabel("Sweetness")
plt.title("SVM - Bananas vs Apples")

plt.legend()
plt.grid()

plt.show()

# dont forget  : pip install scikit-learn matplotlib