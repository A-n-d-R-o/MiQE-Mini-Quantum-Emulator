![WIP](https://img.shields.io/badge/status-WIP-yellow)
# MiQE: Mini Quantum Emulator
Single-file quantum computer emulator. Currently capable of simulating small-scale (below 10 qubits) quantum algorithms and circuits. This easy to use program is intended for use by those wanting to gain a deeper understanding on the logic behind quantum computation, or those wanting to simulate their own quantum systems. The size and limited complexity of the program, MiQE, allows for easy modification of the source code allows for a variety of simple but robust investigations into quantum computation. MiQE uses big-endian notation for qubit ordering.

<br/>
Inspired by Qiskit, though not nearly as powerful, efficient or useful.

<br/>
## Current Features
* Initialisation and conversion between state vector form and density matrix form.
* Single-qubit Pauli error simulation and random error simulation.
* Unary and multi-control qubit gate implementation, with a near-full quantum gate library.
* Partial- and full measurements.
* Simple implementation of the dephasing and depolarising quantum channels (exclusive for density matrix structure).
* Visualisation of state vectors (in column form and Dirac notation) and density matrix through LaTeX formatting.
* Probability amplitude plotting though Matplotlib.
* A function to run quantum circuits and plot basis state measurements through Matplotlib.

<br/>
## Future Plans
* More visualisation tools (graphs, circuit diagrams, Bloch spheres, etc.).
* Make the implementation more efficient
* Extend the functionality and ability of the program (for implementation of more useful algorithms).

