import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

print("=" * 70)
print("OPGAVE 13: Afkøling af ris")
print("=" * 70)


# Differentialligning: dy/dt = -t - 0.05*y
def dydt(y, t):
    return -t - 0.05 * y


# Begyndelsesbetingelse fra opgaven:
# Ved t=0 er dy/dt = -3
# -0 - 0.05*y0 = -3
# y0 = 60 grader C
y0 = 60

print("\nDifferentialligning: dy/dt = -t - 0.05*y")
print(f"Begyndelsesbetingelse: y(0) = {y0}°C")
print("(baseret på at dy/dt = -3 ved t=0)")

# DEL A: Væksthastighed når temperaturen er 70°C
print("\n--- Del a: Væksthastighed ved 70°C ---")
print("Hvis temperaturen er 70°C, afhænger hastigheden af tiden t:")
print("dy/dt = -t - 0.05*70 = -t - 3.5")
print("\nVæksthastighed ved forskellige tidspunkter:")
for t in [0, 5, 10, 15, 20, 25, 30]:
    rate = -t - 0.05 * 70
    print(f"  t = {t:2d} min: dy/dt = {rate:6.2f}°C/minut")

# DEL B: Analytisk løsning
print("\n--- Del b: Analytisk formel for f(t) ---")
print("\nLøsning af differentialligningen dy/dt + 0.05*y = -t")
print("Metode: Integrationsfaktor μ(t) = e^(0.05*t)")
print("\nGenerel løsning: y(t) = -20*t + 400 + C*e^(-0.05*t)")
print(f"Med y(0) = {y0}: C = {y0 - 400}")
print("\nEndelig formel:")
print(f"f(t) = 400 - 20*t + {y0 - 400}*e^(-0.05*t)")
print("eller")
print("f(t) = 400 - 20*t - 340*e^(-0.05*t)")


def f_analytical(t):
    return 400 - 20 * t - 340 * np.exp(-0.05 * t)


# Verificer løsningen
print("\nVerificering:")
print(f"f(0) = {f_analytical(0):.2f}°C (skal være {y0}°C)")

# Beregn afledt numerisk for at verificere
epsilon = 0.001
df_0 = (f_analytical(epsilon) - f_analytical(0)) / epsilon
print(f"f'(0) ≈ {df_0:.2f}°C/minut (skal være -3°C/minut)")

# Vis temperatur over tid
print("\nTemperatur over tid:")
time_values = [0, 10, 20, 30, 40, 50, 60]
for t in time_values:
    print(f"  t = {t:2d} min: f({t}) = {f_analytical(t):7.2f}°C")

# Numerisk løsning for sammenligning
t_span = np.linspace(0, 60, 1000)
y_numerical = odeint(dydt, y0, t_span)

# Plot temperatur over tid
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(t_span, y_numerical, "b-", linewidth=2, label="Numerisk løsning")
plt.plot(t_span, f_analytical(t_span), "r--", linewidth=2, label="Analytisk løsning")
plt.axhline(y=70, color="g", linestyle=":", label="y = 70°C")
plt.xlabel("Tid (minutter)", fontsize=12)
plt.ylabel("Temperatur (°C)", fontsize=12)
plt.title("Opgave 13: Afkøling af ris", fontsize=14, fontweight="bold")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 60)

# Plot væksthastighed
plt.subplot(1, 2, 2)
dy_dt_values = np.array([-t - 0.05 * 70 for t in t_span])
plt.plot(t_span, dy_dt_values, "purple", linewidth=2)
plt.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
plt.xlabel("Tid (minutter)", fontsize=12)
plt.ylabel("Væksthastighed (°C/minut)", fontsize=12)
plt.title("Væksthastighed ved y = 70°C", fontsize=14, fontweight="bold")
plt.grid(True, alpha=0.3)
plt.xlim(0, 60)

plt.tight_layout()
plt.savefig("opgave_13_afkoling.png", dpi=300, bbox_inches="tight")
print("\nGraf gemt som 'opgave_13_afkoling.png'")

# =============================================================================
print("\n\n" + "=" * 70)
print("OPGAVE 10: Vektorfunktion og banekurve")
print("=" * 70)


# Vektorfunktion s(t) = (1/4*t^4 + 2t, t^3 - t + 2)
def s(t):
    x = 0.25 * t**4 + 2 * t
    y = t**3 - t + 2
    return x, y


# Hastighedsvektor s'(t) = (t^3 + 2, 3t^2 - 1)
def velocity(t):
    vx = t**3 + 2
    vy = 3 * t**2 - 1
    return vx, vy


print("\nVektorfunktion: s⃗(t) = (1/4*t⁴ + 2t, t³ - t + 2)")
print("Hastighedsvektor: s⃗'(t) = (t³ + 2, 3t² - 1)")

# DEL B: Vandret tangent
print("\n--- Del b: Vandret tangent ---")
print("Vandret tangent ⟺ y-komponent af hastighed = 0")
print("3t² - 1 = 0")
print("t² = 1/3")
print("t = ±√(1/3) = ±√3/3")

t1 = np.sqrt(1 / 3)
t2 = -np.sqrt(1 / 3)

print(f"\nt₁ = {t1:.6f} ≈ {t1:.3f}")
print(f"t₂ = {t2:.6f} ≈ {t2:.3f}")

p1_x, p1_y = s(t1)
p2_x, p2_y = s(t2)

print(f"\nPunkt P₁ ved t = {t1:.3f}: ({p1_x:.3f}, {p1_y:.3f})")
print(f"Punkt P₂ ved t = {t2:.3f}: ({p2_x:.3f}, {p2_y:.3f})")

# Verificer at hastighederne er vandrette
v1_x, v1_y = velocity(t1)
v2_x, v2_y = velocity(t2)

print(f"\nHastighed ved P₁: ({v1_x:.3f}, {v1_y:.6f})")
print(f"Hastighed ved P₂: ({v2_x:.3f}, {v2_y:.6f})")
print("✓ Y-komponenterne er ~0, så tangenterne er vandrette")

# DEL A: Tegn banekurven
print("\n--- Del a: Tegn banekurven ---")

# Generer punkter for banekurven
t_values = np.linspace(-2, 2, 200)
x_values = []
y_values = []

for t in t_values:
    x, y = s(t)
    x_values.append(x)
    y_values.append(y)

x_values = np.array(x_values)
y_values = np.array(y_values)

print(f"Genereret {len(t_values)} punkter for banekurven")
print(f"x-interval: [{x_values.min():.2f}, {x_values.max():.2f}]")
print(f"y-interval: [{y_values.min():.2f}, {y_values.max():.2f}]")

# Nøglepunkter
key_t_values = [-2, -1, t2, 0, t1, 1, 2]
print("\nNøglepunkter på banekurven:")
for t_val in key_t_values:
    x, y = s(t_val)
    note = (
        " (Vandret tangent)" if np.isclose(t_val, t1) or np.isclose(t_val, t2) else ""
    )
    print(f"  t = {t_val:6.3f}: ({x:7.3f}, {y:7.3f}){note}")

# Plot banekurven
plt.figure(figsize=(10, 8))

# Tegn banekurven
plt.plot(x_values, y_values, "b-", linewidth=2.5, label="Banekurve")

# Marker startpunkt
x_start, y_start = s(-2)
plt.plot(x_start, y_start, "go", markersize=12, label="Start (t=-2)", zorder=5)

# Marker slutpunkt
x_end, y_end = s(2)
plt.plot(x_end, y_end, "ro", markersize=12, label="Slut (t=2)", zorder=5)

# Marker punkter med vandret tangent
plt.plot(
    p1_x,
    p1_y,
    "orange",
    marker="o",
    markersize=15,
    label=f"Vandret tangent (t≈{t1:.3f})",
    zorder=5,
)
plt.plot(
    p2_x,
    p2_y,
    "orange",
    marker="o",
    markersize=15,
    label=f"Vandret tangent (t≈{t2:.3f})",
    zorder=5,
)

# Tegn vandrette tangenter
tangent_length = 1.5
plt.plot(
    [p1_x - tangent_length, p1_x + tangent_length],
    [p1_y, p1_y],
    "orange",
    linestyle="--",
    linewidth=2,
)
plt.plot(
    [p2_x - tangent_length, p2_x + tangent_length],
    [p2_y, p2_y],
    "orange",
    linestyle="--",
    linewidth=2,
)

# Tegn hastighedsvektorer ved nogle punkter
arrow_t_values = [-1.5, -0.5, 0.5, 1.5]
scale = 0.3

for t_val in arrow_t_values:
    x, y = s(t_val)
    vx, vy = velocity(t_val)
    plt.arrow(
        x,
        y,
        vx * scale,
        vy * scale,
        head_width=0.15,
        head_length=0.1,
        fc="purple",
        ec="purple",
        alpha=0.6,
        width=0.02,
    )

# Akser og labels
plt.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
plt.axvline(x=0, color="gray", linestyle="-", linewidth=0.5)
plt.xlabel("x", fontsize=14)
plt.ylabel("y", fontsize=14)
plt.title(
    "Opgave 10: Banekurve s⃗(t) = (¼t⁴ + 2t, t³ - t + 2)", fontsize=14, fontweight="bold"
)
plt.legend(loc="best", fontsize=10)
plt.grid(True, alpha=0.3)
plt.axis("equal")

plt.tight_layout()
plt.savefig("opgave_10_banekurve.png", dpi=300, bbox_inches="tight")
print("\nGraf gemt som 'opgave_10_banekurve.png'")

plt.show()

print("\n" + "=" * 70)
print("FÆRDIG! Graferne er gemt og vises nu.")
print("=" * 70)
