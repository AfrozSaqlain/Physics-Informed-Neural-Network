# Physics-Informed Neural Networks (PINNs)

This repository contains implementations of **Physics-Informed Neural Networks (PINNs)** applied to various problems in physics. Each notebook demonstrates how PINNs can be used to solve differential equations and extract meaningful physical insights.

## Notebooks Overview

- **[`PINN.ipynb`](PINN.ipynb)**: Provides a general introduction to **Physics-Informed Neural Networks**, explaining their core concepts and applications.  
- **[`TISE.ipynb`](TISE.ipynb)**: Implements a PINN model to solve the **Time-Independent Schrödinger Equation (TISE)**. Given an arbitrary potential, the model predicts the corresponding **eigenfunctions** and **eigenvalues**.  
- **[`QNM_Kerr_BH_Animated.ipynb`](QNM_Kerr_BH_Animated.ipynb)**: Uses a PINN-based approach to solve the **Teukolsky equation**, extracting **Quasi-Normal Modes (QNMs)** of a Kerr black hole. The notebook includes an animated visualization of the solutions.  
- **[`PM_Amplification_Factor.ipynb`](PM_Amplification_Factor.ipynb)** *(Work in Progress)*: Aims to compute the **amplification factor due to gravitational lensing**, leveraging PINNs to model the lensing effects on propagating waves.  

Feel free to explore and contribute!
