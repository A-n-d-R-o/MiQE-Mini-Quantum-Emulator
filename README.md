![WIP](https://img.shields.io/badge/status-WIP-yellow)

# MiQE: Mini Quantum Emulator

**Single-file quantum computer emulator.**  
Currently capable of simulating small-scale (below 10 qubits) quantum algorithms and circuits.

This easy-to-use program is intended for:
- Learners wanting to understand the logic behind quantum computation.
- Users who want to simulate their own simple quantum systems.

The compact and low-complexity design of **MiQE** makes it easy to modify and explore. It supports a variety of simple but robust quantum computing investigations.  
**Note:** MiQE uses big-endian notation for qubit ordering.

> Inspired by Qiskit — though not nearly as powerful, efficient, or full-featured.

---

## ✅ Current Features

- Initialization and conversion between state vector and density matrix forms.
- Single-qubit Pauli error simulation and random error injection.
- Unary and multi-control qubit gate implementation, with a near-complete quantum gate library.
- Partial and full measurements.
- Simple dephasing and depolarizing channels (exclusive to density matrices).
- Visualization of:
  - State vectors (column form and Dirac notation)
  - Density matrices (via LaTeX formatting)
- Probability amplitude plotting using Matplotlib.
- Circuit execution with basis state measurement plots.

---

## 🧭 Future Plans

- More visualization tools (circuit diagrams, Bloch spheres, graphs, etc.).
- Improve efficiency and performance.
- Expand algorithm support and extend functionality.
