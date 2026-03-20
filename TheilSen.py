import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, TheilSenRegressor

# Beispieldaten
x = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y = np.array([20, 22, 24, 26, 100, 30])

# Normale lineare Regression
lin_reg = LinearRegression()
lin_reg.fit(x, y)
y_lin = lin_reg.predict(x)

# Robuste Regression mit Theil-Sen
theil_sen = TheilSenRegressor(random_state=42)
theil_sen.fit(x, y)
y_theil = theil_sen.predict(x)

# Ausgabe der Gleichungen
print("Normale Regression:")
print(f"Steigung: {lin_reg.coef_[0]:.2f}")
print(f"Achsenabschnitt: {lin_reg.intercept_:.2f}")

print("\nTheil-Sen Regression:")
print(f"Steigung: {theil_sen.coef_[0]:.2f}")
print(f"Achsenabschnitt: {theil_sen.intercept_:.2f}")

# Plot
plt.scatter(x, y, label="Messwerte")
plt.plot(x, y_lin, label="Normale Regression")
plt.plot(x, y_theil, label="Theil-Sen Regression")
plt.xlabel("Zeit")
plt.ylabel("Temperatur")
plt.title("Vergleich: normale Regression vs. Theil-Sen")
plt.legend()
plt.grid(True)
plt.show()