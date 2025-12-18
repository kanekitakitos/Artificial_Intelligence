# 🧠 Artificial Intelligence – Java Laboratories

[![Java](https://img.shields.io/badge/Java-17%2B-orange.svg)](https://www.oracle.com/java/technologies/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-In%20Development-yellow.svg)]()

> This repository presents a collection of laboratory implementations of fundamental Artificial Intelligence algorithms, developed in Java, with a primary focus on heuristic methods and artificial neural networks implemented entirely from first principles.

## 🎯 Overview

This repository contains a set of practical laboratory exercises designed to explore core concepts in Artificial Intelligence. Each laboratory addresses a specific topic, ranging from heuristic-based problem solving to the mathematical formulation and implementation of artificial neural networks, including Perceptrons and Multi-Layer Perceptrons (MLPs).

All algorithms are implemented *from scratch*, without the use of external machine learning or numerical computation frameworks, with the objective of providing a rigorous understanding of the underlying mathematical models and computational procedures.

## 🚀 Available Laboratories

### 🧩 [Lab 00: Small Instances of the 8-puzzle](00%20-%20Small%20Instances%20of%20the%208-puzzle/)
**Introduction to state-space search using the 8-puzzle**

- **Scope**: Solving 3x3 sliding tile puzzles.
- **Topics Covered**:
    - State representation (`Board` class)
    - Implementation of the `Ilayout` interface
    - Best-First Search algorithm (Uniform-Cost Search variant)

### 🔢 [Lab 01: Problem Array Sorting](01%20-%Problem%20Array%208Sorting/)
**Sorting arrays via Uniform-Cost Search**

- **Scope**: Finding the optimal sequence of swaps to sort an array with variable swap costs.
- **Topics Covered**:
    - Abstract search architecture (`AbstractSearch` template)
    - Uniform-Cost Search (`GSolver`)
    - Cost function implementation for state transitions

### 🧩 [Lab 02: Problem Heuristics](02%20-%20Problem%20Heuristics%20for%20the%20Array%20Sorting/)
**Application of heuristic methods to sorting and search problems**

- **Scope**: Heuristic search strategies and array sorting problems.
- **Topics Covered**:
    - Formal definition of the state space
    - Design and implementation of heuristic evaluation functions
    - Comparative analysis of heuristic-based sorting and search approaches
    - Computational complexity considerations

### 🤖 [Lab 03: Perceptrons](03%20-%20Perceptrons/)
**Implementation of the single-layer neural network model**

- **Core Component**: `Matrix.java`
- **Topics Covered**:
    - **Custom Mathematical Library**: Implementation of matrix operations, including dot products, transposition, and element-wise transformations
    - **Artificial Neuron Model**: Representation of synaptic weights, bias, and linear activation
    - **Linear Classification Tasks**: Resolution of classical problems such as logical AND and OR gates
    - **Learning Algorithm**: Supervised training using the Perceptron learning rule

**Example Usage (Matrix Operations)**:
```java
// Definition of input matrices
Matrix input = new Matrix(new double[][]{{0, 1}, {1, 0}});
Matrix weights = Matrix.Rand(2, 1, 12345);

// Feedforward computation
Matrix output = input.dot(weights);
```

### 👁️ [Lab 04: MLP – Identification of Digits 3 and 2](04%20-%20MLP%20identify%203%20and%202/)
**Multi-Layer Perceptron applied to computer vision**

- **Scope**: Binary classification of handwritten digits (3 versus 2).
- **Topics Covered**:
    - **Feedforward Architecture**: Neural networks composed of multiple hidden layers
    - **Backpropagation Algorithm**: Gradient-based error propagation for weight optimization
    - **Activation Functions**: Non-linear transformations (e.g., Sigmoid) applied via `Matrix.apply()`
    - **Model Validation**: Performance evaluation using a subset of image data

## 🛠️ Requirements and Installation

### Prerequisites
- Java Development Kit (JDK) version 17 or higher
- Integrated Development Environment (IDE), such as IntelliJ IDEA, Eclipse, or Visual Studio Code

### Installation Procedure
```bash
# Clone the repository
git clone https://github.com/your-username/Artificial_Intelligence.git

# Navigate to the project directory
cd Artificial_Intelligence

# Open the project using the preferred IDE
```

## 🔧 Project Structure

```
Artificial_Intelligence/
├── 02 - Problem Heuristics for the Array Sorting/
│   └── ...
├── 03 - Perceptrons/
│   ├── src/
│   │   └── math/
│   │       └── Matrix.java
│   └── ...
├── 04 - MLP identify 3 and 2/
│   └── ...
├── src/
│   └── ...
└── README.md
```

## 📚 Documentation and Methodology

The laboratories emphasize a *from-scratch* methodology, allowing a detailed examination of the mathematical foundations of Artificial Intelligence algorithms:
- **Linear Algebra**: Explicit implementation and manipulation of matrices and vectors
- **Calculus**: Computation of gradients and partial derivatives for learning algorithms, particularly backpropagation

## 🧪 Experimental Evaluation and Validation

The implemented algorithms were evaluated and validated through:
- **Canonical Problems**: Logical gate classification (AND, OR, XOR) for Perceptron and MLP models
- **Real-World Datasets**: Handwritten digit classification tasks (digits 2 and 3)
- **Mathematical Verification**: Unit testing of matrix operations to ensure numerical correctness

## 🎓 Educational and Academic Relevance

This repository is particularly suitable for:
- **Students of Artificial Intelligence and Machine Learning**: As a didactic resource for understanding the internal mechanisms of neural networks
- **Computer Science and Software Engineering Courses**: As an example of structured implementation of mathematical algorithms

## 📄 License

This project is distributed under the MIT License.

## 👨‍💻 Author

Brandon Mejia
