---
title: 'Integration with Newton-Cotes and Gauss-quadrature'
date: 2025-02-12
permalink: /posts/2025/02/Integration/
---
In many real-world problems, we need to approximate integrals because finding an exact solution is too complex or even impossible. Methods like `Newton-Cotes` (Closed and Open) and `Gauss Quadrature` help us solve these integrals efficiently. 


Like all numeric methods, we are trying to do some "tricks" and approcimations to still manage to solve complex or unknown function. for this time we'll focues on Newton-Cotes and Gauss-quadrature for Integration and Interpolation.

# Why do we need those methods? 
---
Lets say for a real world exapmle - we want to calculte the pollution spread in a river and to estimat the total pollution load in a river over a given section.
to talk about the sum over time or over space is takeing us to use integrals. 

for smlisicy lets use the mass transported where $C(x)$ is the pollution concentraion at location $x$ and $V(x)$ is the flow velocity.

Now, $C(x)$ and $V(x)$ are measured at discrete locations, that is to say we do not have the whole domain meathed at all locastion (which would be not efficent this to do)

our integral would be: 
$$M = \int_0^L C(x) V(x) \,dx$$

where:

- $M$ = Total pollutant mass transported (e.g., kg/day)
- $C(x)$ = Pollutant concentration at location $x$ (e.g., mg/L)
- $V(x)$ = Flow velocity at location $x$ (e.g., m/s)
- $x$ = Position along the river (e.g., meters)
- $L$ = Total river length considered (e.g., km)

real-world hydrology and fluid dynamics equations that describe pollutant transport can be complicated. foe example 

## **Complex Models Requiring Numerical Integration**

Some environmental models require **numerical integration** due to their complexity, including **partial differential equations (PDEs), non-linearity, or spatially varying parameters**. Below are three such models:

---

## **1️⃣ Advection-Diffusion Equation (ADE)**
The **Advection-Diffusion Equation (ADE)** describes pollutant transport in flowing water:

$$ \frac{\partial C}{\partial t} + V \frac{\partial C}{\partial x} = D \frac{\partial^2 C}{\partial x^2} - k C + S(x,t) $$

where:
- $C(x,t)$ = Pollutant concentration (mg/L)
- $V(x)$ = Flow velocity (m/s)
- $D(x)$ = Diffusion coefficient (m²/s)
- $k$ = Decay rate of pollutant (1/s)
- $S(x,t)$ = External source term (e.g., industrial discharge)


- This is a **partial differential equation (PDE)** requiring **numerical solvers**.
- When parameters **$V(x)$, $D(x)$, and $S(x,t)$** vary along the river

---

## **2️⃣ Darcy’s Law (Groundwater Exchange with River)**
Groundwater inflow can introduce or dilute pollutants in rivers. The **exchange flow** is described by **Darcy’s Law**:

$$ Q_g = -K A \frac{dh}{dx} $$

where:
- $Q_g$ = Groundwater discharge into the river (m³/s)
- $K$ = Hydraulic conductivity (m/s)
- $A$ = Cross-sectional area of exchange (m²)
- $\frac{dh}{dx}$ = Hydraulic gradient

**Why Numerical Integration?**
- This is an **ordinary differential equation (ODE)** but becomes **nonlinear** when parameters like **$K(x)$** or **$A(x)$** change along the river.

---

These models are too complex for simple analytical solutions. **Newton-Cotes (Simpson’s Rule) and Gauss-Quadrature** provide efficient numerical methods for solving these integrals over real-world river systems.

>[!info]
>Note this method is for 1D, but assuming the main flowing is downsteam we could still benifit from it.
# How to use Newton-Cotes method - closed and open

Now that we convinsed it would be more afficent to use nomeric methods for this kind of problem, let see how we do it.

## Closed Newton-Cotes
**Closed Newton-Cotes** is a numerical integration method that approximates an integral using **equally spaced and fixed points**, including **both endpoints** of the interval [a,b].
the intgral is approximation the sun of weighted function evaluations at those points we have.

if we have 2 points we wolud usse Linear approximation - which is not very accorate

if we have 3 points we would use parabola - Simpson's Role

### what is Sompson's Role?
Simpson’s Rule is a **numerical integration method** that approximates an integral by fitting a **parabola** through three equally spaced points, including the **endpoints**.

# **Simpson’s Rule (Closed Newton-Cotes, 3 Points)**  

Simpson’s Rule is a **numerical integration method** that approximates an integral by fitting a **parabola** through three equally spaced points, including the **endpoints**.

## **Formula**  
For an integral:

$$
I = \int_a^b f(x) \,dx
$$

Simpson’s Rule approximates it as:

$$
I \approx \frac{h}{3} \left[ f(x_0) + 4 f(x_1) + f(x_2) \right]
$$

where:
- $x_0 = a$ (start point),
- $x_1 = \frac{a+b}{2}$ (midpoint),
- $x_2 = b$ (end point),
- $h = \frac{b - a}{2}$ (step size).

> [!Note]
> those wights of 1, 4, 1 are determent by Lagrange basis polynomials e.g. the wieghts for quadratic polynomial is given more wight to the midpoint because it gives more information abut the domin.

> [!important]
> the Polynomial degree is depends on the number of points and **not** by the function it self. what is to say it is better if we have n+1 points, but is we do not we can still apply the method but to expext higher error

# **Derivation of Simpson’s Rule Error Term**
To quantify the error, we use **Taylor series expansion** of $f(x)$ around the midpoint $x1$:


Using **Taylor series expansion**, a function $f(x)$ around the midpoint $x_1$ can be written as:

$$
f(x) = f(x_1) + f'(x_1)(x - x_1) + \frac{f''(x_1)}{2!} (x - x_1)^2 + \frac{f'''(x_1)}{3!} (x - x_1)^3 + \frac{f''''(c)}{4!} (x - x_1)^4 + \dots
$$

When integrating this expansion over $[a, b]$, the even-degree terms up to $x^2$cancel out perfectly, since the **Simpson’s Rule weights** exactly match the integral of a quadratic function.

The **first non-zero error term** (the first mistake) comes from $x^4$, which involves $f''''(c)$, leading to the error formula:

$$
E = -\frac{h^5}{90} f''''(c)
$$

where $c$ is some unknown point in $(a, b)$, so we use the max option to find the max error. 


# Lets go back to our river problem
Again, assuming we only solving for 1D, now it can be that we can't get measurmennt is the exact start and end point, maybe because of trubulent or some other issues like rocks, vegetations and so on.

this means we nned to ignore the start and end points snd use only the interior points.

now that we understand the problem, let's solve it

# **Solving Using Open Newton-Cotes**

To estimate the total pollution transport in a river without using endpoint measurements, we apply the **3-point Open Newton-Cotes rule**:

$$
M \approx \frac{4h}{3} \left[ 2f(x_1) - f(x_2) + 2f(x_3) \right]
$$

where:

- $h$ is the step size between consecutive points:

$$
h = \frac{x_3 - x_1}{3} 
$$

The weight here are different compared to the closed approach, in the closed method we used 1, 4, 1 which gave more wight to the inner point, because we could drive more information from it. But here for the open method, it is the opposite - the wights are 2, -1, 2.

# Gauss Quadrature

Gauss Quadrature picks the best points inside the interval, not equally spaced, unlike Newton-Cotes where points are picked evenly.

those 'best points' are the **roots of Legendre polynomials**.

For Gauss Quadratue the sampling points are not equally-spaced, but in our case for 3 points, it is evenly spread. 

The nodes are chosen from [-1,1] because Gauss-Legendre Quadrature is optimized for this range and We need to change the integration boundary over [0,2], the nodes transform accordingly. Exact solution for polynomials up to  2𝑛+2.

# **Summary: Gauss Quadrature Weights**  

Gauss Quadrature approximates an integral using weighted function values at optimized points:

$$
I \approx \sum_{i=1}^{n} w_i f(x_i)
$$

where:
- $x_i$ are the **Legendre polynomial roots**.
- $w_i$ are the **weights**, computed as:

$$
w_i = \int_{-1}^{1} \prod_{j \neq i} \frac{x - x_j}{x_i - x_j} dx
$$

### **Key Facts:**
- **Weights depend only on $n$, not on $f(x)$.**
- **Once $n$ is chosen, $x_i$ and $w_i$ are fixed.**
- **Gauss Quadrature is more accurate than Newton-Cotes for the same $n$.**

Example for **3-point Gauss Quadrature**:

| $x_i$ | $w_i$ |
|------|------|
| $-0.7746$ | $0.5556$ |
| $0$ | $0.8889$ |
| $0.7746$ | $0.5556$ |


