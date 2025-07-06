# 🎵 LSTMiniNet: Custom LSTM Melody Generator in Modern C++

**LSTMiniNet** is a handcrafted LSTM-based melody generator built entirely in modern C++20, powered by a custom matrix library and a reverse-mode automatic differentiation engine. This project demonstrates building deep sequence models like LSTM from scratch, without any external ML/DL libraries.

---

## 📌 Features

* 🚀 Pure C++ LSTM architecture using manually defined computation graphs
* 🧠 Gradient backpropagation via a custom Reverse AutoDiff Engine
* 🧮 Matrix operations module with support for activation functions, softmax, and loss functions
* 🎹 Generates music note sequences using an LSTM trained on melody data
* 📉 Support for loss functions like MSE, BCE, CCE
* 🔄 Truncated BPTT and gradient clipping for stability

---

## 🧠 Architecture Overview

### 🧮 Reverse AutoDiff Engine

* All operations (addition, multiplication, activation, etc.) are captured dynamically via `shared_ptr<Node>`
* Each `Node` stores the gradient and defines a `backward()` lambda to propagate gradients to its parents

```cpp
shared_ptr<Node> A, B;
shared_ptr<Node> C = A * B; // Gradient tracked automatically
```

### 🧩 Matrix Module

* Core data structure for computations
* Supports broadcasting, elementwise ops, transpose, activation derivatives
* Integrated with autodiff via operator overloading

### 🔁 LSTM Cell Logic

Each LSTM cell performs 4 computations:

1. **Forget Gate:** `Ft = sigmoid(Wf * xt + Uf * ht-1 + bf)`
2. **Input Gate:** `It = sigmoid(Wi * xt + Ui * ht-1 + bi)`
3. **Candidate Memory:** `Ct~ = tanh(Wc * xt + Uc * ht-1 + bc)`
4. **Cell State Update:** `Ct = Ft ⊙ Ct-1 + It ⊙ Ct~`
5. **Output Gate:** `Ot = sigmoid(Wo * xt + Uo * ht-1 + bo)`
6. **Hidden State:** `Ht = Ot ⊙ tanh(Ct)`

---

## 📈 Training Details

* Dataset: Melody note sequences (`Melody.txt`)
* Input: sequence of notes → Output: next note prediction
* Training: Sequence-to-sequence LSTM using CrossEntropyLoss
* Optimization: Manual SGD + L2 regularization
* Extras: Gradient clipping, weight decay, truncation step

```cpp
model.Train(dataset, targets, epochs = 4, lr = 0.05, clip = 1.0, L2 = 0.05, truncate = 3);
```

---

## 🎼 Output Sample

Once trained, the model can generate sequences:

```cpp
vector<int> seed = dataset.back();
Matrix nextMelody = model.predict(seed);
```

### 🖼️ Output

<img src="Output.png" width="700" />

This shows a printed sequence of generated notes, predicted one-by-one, simulating a musical continuation.

---

## 🧮 GRU Gate Comparison (Not Used in Code)

While the project uses LSTM, here's how GRU differs:

* **Update Gate:** Controls memory retention
* **Reset Gate:** Controls forgetting of past hidden state
* **Fewer parameters** than LSTM (no separate cell state)

**GRU Equations:**

```
zₜ = sigmoid(Wz * xₜ + Uz * hₜ₋₁)
rₜ = sigmoid(Wr * xₜ + Ur * hₜ₋₁)
ĥₜ = tanh(W * xₜ + U * (rₜ ⊙ hₜ₋₁))
hₜ = (1 - zₜ) ⊙ hₜ₋₁ + zₜ ⊙ ĥₜ
```

### 🖼️ GRU Architecture

<img src="GRU-RNN.png" width="650" />

---

## 🧠 LSTM Architecture

<img src="LSTM-RNN.png" width="700" />

This shows the flow of data through input, forget, and output gates and how memory is updated and passed.

---

## 📃 License

MIT License © 2025 MAURYA

---

## 💬 Author Note

This project is part of a larger initiative to build deep learning models from scratch in C++, without relying on external frameworks. Stay tuned for more architectures like GRU, Transformer, and full sequence-to-sequence models!
