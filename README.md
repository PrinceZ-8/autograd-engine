# Autograd Engine

A lightweight **reverse-mode automatic differentiation engine** built from scratch using **Python and NumPy**.

This project demonstrates the fundamental concepts behind automatic differentiation and backpropagation—the same core ideas used by modern machine learning frameworks.

## 🚀 Features

- Reverse-mode automatic differentiation
- Computational graph construction
- Automatic gradient calculation using backpropagation
- Gradient accumulation
- Scalar and matrix operations
- Built using Python and NumPy
- Lightweight and easy to understand
- Educational implementation of core autograd concepts

## 🧠 How It Works

Consider the mathematical expression:

```text
z = x * y + x
```

The engine constructs a computational graph:

```text
x ──┐
    × ──┐
y ──┘   │
        + ── z
x ──────┘
```

During the **forward pass**, the engine calculates the output value.

During the **backward pass**, it traverses the computational graph in reverse order and applies the **chain rule** to calculate gradients.

## 📦 Requirements

- Python 3.x
- NumPy

Install the required dependency:

```bash
pip install numpy
```

## 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/PrinceZ-8/autograd-engine.git
cd autograd-engine
```

## ▶️ Running the Project

Run the main Python file:

```bash
python engine.py
```

## 🔄 Reverse-Mode Automatic Differentiation

Reverse-mode automatic differentiation is particularly useful when there are many input parameters and a single output, such as a loss function.

```text
Parameters
     ↓
Neural Network
     ↓
    Loss
```

The autograd engine computes the gradient of the loss with respect to each parameter:

```text
∂Loss/∂Parameter
```

These gradients can then be used by optimization algorithms such as gradient descent.

## 📚 Concepts Covered

This project helps demonstrate and explain:

- Computational graphs
- Automatic differentiation
- Reverse-mode differentiation
- The chain rule
- Forward propagation
- Backpropagation
- Gradient calculation
- Gradient accumulation

## 🧪 Example Workflow

```text
1. Create input values
        ↓
2. Perform mathematical operations
        ↓
3. Build the computational graph
        ↓
4. Compute the forward result
        ↓
5. Run backward propagation
        ↓
6. Retrieve gradients
```

## 📁 Project Structure

```text
autograd-engine/
│
├── engine.py        # Core automatic differentiation engine
└── README.md        # Project documentation
```

## 🎯 Future Improvements

Potential enhancements include:

- Additional mathematical operations
- Activation functions such as ReLU, Sigmoid, and Tanh
- Broadcasting support
- Neural network layers
- Loss functions
- Optimizers such as SGD and Adam
- Gradient checking
- Unit tests
- Performance improvements

## 🤝 Contributing

Contributions and improvements are welcome!

```bash
git checkout -b feature/my-feature
git commit -m "Add my feature"
git push origin feature/my-feature
```

Then open a Pull Request.

## 📄 License

This project does not currently specify a license. Consider adding an MIT License or another open-source license if you plan to make the project publicly reusable.

## ⭐ Purpose

The goal of this project is to provide a simple, understandable implementation of an **autograd engine from scratch**.

It is designed for developers, students, and machine learning enthusiasts who want to understand what happens internally when frameworks automatically calculate gradients during neural network training.

---

⭐ If you find this project useful, consider giving the repository a star!
