## Contents

 - [Required Packages](#required-packages)

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
* `xyz_errors` (list[float, float, float]): a list of 3 floats, ranging from 0 to 1. Respectively, these numbers represent the error rates for Pauli-X, Pauli-Y, and Pauli-Z errors acting on qubits during implementation of operations and measurement gates.
* `random_error ` (float): float ranging from 0 to 1, specifying the random noise rate present in the system, it is more general than the `xyz_errors` input parameter.

The error parameters default to 0. No noise is simulated unless specified.

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

The `dephase` channel introduces noise by zeroing out off-diagonal entries of the density matrix. The magnitude of the zeroing out is defined by epsilon. Essentially eliminates superposition, turning the system into a classical probabilistic state.

* `*qubits` (int): the qubits to apply dephasing to.
* `epsilon` (float): ranging from 0 to 1, this defines how zeroed out the off-diagonal entries should be, where 0 means no dephasing and 1 (default) means full dephasing (density matrix is turned into a diagonal matrix).

```python
def depolarise(self, *qubits, epsilon=1.0):
```

The `depolarise` channel destroys the quantum information contained within a qubit by transforming it into the completely mixed state.

**Note:** after applying this gate, the structure cannot be changed to `'StateVector'` and the system cannot be measured (this is a major limitation in the usefulness of this method).

* `*qubits` (int): the qubits to apply depolarisation to.
* `epsilon` (float): ranging from 0 to 1, this defines how destroyed the information held by the qubit should be. If 0, then no destruction occurs, if 1 (default), then all information is lost.

### Measurement Methods

```python
def measure(self, *qubits, collapse=True):
```

Implement partial measurement on the system.

* `*qubits` (int): the qubit indices to measure.
* `collapse` (bool): if True (default), then the system retains its measured state (it stays measured), if False then the system does not evolve into the measured state, but retains its pre-measured state.

```python
def measure_all(self, collapse=True):
```

Implements full measurement of the quantum system.

* `collapse` (bool): if True (default), then the system evolves into the measured standard basis state, if False, then the system retains its pre-measured state.

### Transformation Methods

```python
def to_density_matrix(self, permanent=False):
```

Converts a `'StateVector'` structure into a `'DensityMatrix'` structure.

* `permanent` (bool): if True, the system stays as a `'DensityMatrix'`, if False (default), then the system does not change after executing this method.

```python
def to_state_vector(self, permanent=False):
```

Converts a `'DensityMatrix'` structure into a `'StateVector'` structure.

* `permanent` (bool): if True, the system stays as a `'StateVector'`, if False (default), then the system does not change after executing this method.

### Visualisation Methods

```python
def show(self, latex=False)
```

Displays the current state of the system, as either a column vector for `'StateVector'` structures or as a density matrix for `'DensityMatrix'` structures.

* `latex` (bool): if True, then the method returns a formatted and clean LaTeX output, if False (default) then the output is a regular Python object.

```python
def diracify(self):
```

Only for `'StateVector'` structures. This transforms the column vector into Dirac notation.

```python
def plot_probs(self, output='list', dims=[6.4, 4.8], x_rot=0, by_qubit=False):
```

Plots the probability amplitudes associated with each basis state of the current quantum system.

* `output` (str or list[str]): the output format. if `'list'` (default) returns a list of the all basis states with their corresponding probability of being measured. If `'plot'` returns a Matplotlib bar chart of the same information.
* `dims` (list[float, float]): the width and height of the plot. Defaulted to the default Matplotlib dimensions
* `x_rot` (float): the rotation (in degrees) of the xtick marks/labels. This is the same as in Matplotlib. Defaulted to 0.
* `by_qubit` (bool): if False (defualt) returns information about the probabilities associated with standard basis states. If True, returns information on the probability of measuring each qubit index, irrespective of the other outcome states.

### Helper Methods

These methods are not to be accessed to called by the user; they're private methods to make the code more compact.

```python
def __apply_gate(self, gate):
```

Applies the full-circuit gate to the quantum system.

* `gate` (matrix): the full-circuit gate, created by taking the Kronecker product between relevant matrices.

```python
@staticmethod
def __clean_format(element):
```

Makes defualt numpy complex numbers more pleasing on the eye.

* `element` (complex float): the complex number to tidy up.

---

## `run_circuit` Function

```python
def run_circuit(circuit, shots=1, output='list', dims=[6.4, 4.8], x_rot=0, by_qubit=False):
```

Runs a quantum circuit (built as a function) a set number of times and outputs information on the measurement results.

* `circuit` (Callable): the quantum circuit function you wish to run. **Note:** `run_circuit` is designed to run quantum circuit which simulate noise, hence why the circuit needs to be rebuilt every time. If you wish to run a noiseless circuit, it would be more efficient to loop over the `measure_all(False)` method and store the data yourself.
* `output` (str or list[str]): the output format. if `'list'` (default) returns a list of the all measured basis states and the number of times that state was measured. If `'plot'` returns a Matplotlib bar chart of the same information.
* `dims` (list[float, float]): the width and height of the plot. Defaulted to the default Matplotlib dimensions
* `x_rot` (float): the rotation (in degrees) of the xtick marks/labels. This is the same as in Matplotlib. Defaulted to 0.
* `by_qubit` (bool): if False (defualt) returns information about the outcome measurements associated with standard basis states. If True, returns information on the number of time each qubit index was measured in the 0 or 1 state.
