## Required Packages

* numpy: for vast majority of mathematical functions and numbers.
* `numpy.linalg`: for linear algebra, including vector norm and eigensystem calculation.
* `functools.reduce`: for taking the Kronecker product between matrices in gate paths.
* `IPython.display.display` and `IPython.display.Math`: formatting and displaying LaTeX visualisation.
* `matplotlib.pyploy`: for clean visualisation and plotting tools.

---

## Gate Library

The base code includes 14 unary qubit gates (4 of which are parameterised). more can be added at will, of course. Below is some of the gates which are included and how they are defined:

```python
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def P(phi):
    return np.array([
        [1, 0], 
        [0, np.exp(1j * phi)]
    ], dtype=complex)
```

---

## `QuantumCircuit` Class
```python
class QuantumCircuit:
```

### Initialiser Method

```python 
def __init__(self, num_qubits, structure='', xyz_errors=[0.0, 0.0, 0.0], random_error=0.0):
```

The method that initialises the quantum system. Where the formalism (hereinafter structure), either state vector or density matrix, is specified, as well as the number of qubits in the system and what, if any noise, is present.

* `num_qubits` (int): the number of qubits to be emulated, given the limitation of the Python language and memory size, this number will have a low upper limit, depending on the system which the program is on.
* `structure` (str or None): if None then it automatically sets the structure to `'StateVector'` (a column vector representing each basis state with probability amplitudes), using `'StateVector'` allows for Dirac notation simulation and faster code. If `'DensityMatrix'`, then a density matrix structure will be used (matrix produced from taking the outer product of a state vector with itself). Using `'DensityMatrix'` allows for channel implementation.
* `xyz_errors` (list[float]): a list of 3 floats, ranging from 0 to 1. Respectively, these numbers represent the error rates for Pauli-X, Pauli-Y, and Pauli-Z errors acting on qubits during implementation of operations and measurement gates.
* `random_error ` (float): float ranging from 0 to 1, specifying the random noise rate present in the system, it is more general than the `xyz_errors` input parameter.

### Noise Methods

```python
def add_gate_errors(self, qubit):
```

This implements the Pauli errors based on the values passed into the `xyz_errors` input parameter for the initialiser method. 

**Note:** although it correctly introduces noise and errors to the system, I cannot say this method has been used in a realistic or accurate way, representative or real quantum hardware.

* `qubit` (int): the qubit index, by big-endian notation, that errors may be introduced to, based on the probabilities passed into the initialiser method.

```python
def add_noise(self):
```

This implements random noise based on the values passed into the `random_error` input parameter for the initialiser method. Like for the `add_gate_errors` method, I am not confident these errors are implemented in a realistic manner.

### Operation Methods

```python
def gate(self, gate, *qubits):
```

Application of a single-qubit, unary, quantum gate.

* `gate` (*gate symbol* str): the name of the gate as seen in the Gate Library, or the parameterised gate along with its rotation angle.
* `*qubits` (int): the specific qubit indices for which the gate acts on.

```python
def C(self, gate, control, target):
```

Application of a (multi-)controlled gate.

* `gate` (*gate symbol* str): the name of the gate to be applied to the target qubit.
* `control` (int or list[int]): the indices of the control qubit(s).
* `target` (int): the index of the target qubit. This cannot be included in the values passed into thr `control` input parameter.

```python
def SWAP(self, q0, q1):
```

Application of the SWAP gate.

* `q0` (int) and `q1` (int): the qubit indices to swap around.

### Channel Methods (only for `'DensityMatrix'` structure)

```python
def dephase(self, *qubits, epsilon=1.0):
```

The `dephase` channel introduces noise by zeroing out off-diagonal entries of the density matrix. The magnitude of the 'zeroing out' is defined by epsilon. Essentially eliminates superposition, turning the system into a classical probabilistic state.

* `*qubits` (int): the qubits to apply dephasing to.
* `epsilon` (float): ranging from 0 to 1, this defines how zeroed out the off-diagonal entries should be, where 0 means no dephasing and 1 means full dephasing (density matrix is turned into a diagonal matrix).

```python
def depolarise(self, *qubits, epsilon=1.0):
```

The `depolarise` channel destroys the quantum information contained within a qubit by transforming it into the completely mixed state.

**Note:** after applying this gate, the structure cannot be changed to `'StateVector'` and the system cannot be measured (this is a major limitation in the usefulness of this method).

* `*qubits` (int): the qubits to apply depolarisation to.
* `epsilon` (float): ranging from 0 to 1, this defines how destroyed the information held by the qubit should be. If 0, then no destruction occurs, if 1, then all information is lost.
