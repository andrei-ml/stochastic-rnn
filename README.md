# Stochastic RNN

This repository provides a PyTorch implementation of a **Stochastic Recurrent Neural Network (SRNN)**. The SRNN extends the standard RNN by introducing stochastic elements, allowing it to model sequential data probabilistically.

In short, the model assumes that the output at time $t$, $\mathbf{y}_t$, follows a multivariate normal distribution:

$$
{\mathbf{y}}_t \sim \mathcal{N}(~\mathbf{f}(\mathbf{x}_{t-1},\ldots,\mathbf{x}_{t-l})~,~\mathbf{\Sigma})
$$


where $\mathbf{f}$ is the RNN function predicting the mean, and $\mathbf{\Sigma}$ is the covariance matrix. Both $\mathbf{f}$ and $\mathbf{\Sigma}$ are learned from the time series. In most cases, $\mathbf{x}$ and $\mathbf{y}$ are the same data, modeling the stochastic dynamics of the observed sequence.

Intuitively, this means the SRNN not only predicts the expected next step in a sequence but also models the uncertainty around that prediction.

---

## 🧪 Features

- **Stochastic RNN** using PyTorch 
- **Utility functions** for loss, training and evaluation  
- **Interactive notebook** for hands-on demonstration of **time series forecasting**
- **Application to atmospheric data** 

---

## 🔧 Installation

To be able to import srnn, clone the repository and install as a package:

```bash
git clone https://github.com/andrei-ml/stochastic-rnn.git
cd stochastic-rnn
pip install .
```

Or, to be able to run the files downloaded here, install the requirements:

```bash
pip install torch numpy scipy
```

To run notebooks, you will also need:

```bash
pip install scikit-learn sdeint
```

---

## 🚀 Quickstart

The easiest way to get started is to run the [Jupyter notebook](quickstart_with_synthetic_example/quickstart.ipynb):

```bash
cd quickstart_with_synthetic_example
jupyter notebook quickstart.ipynb
```

The notebook demonstrates:
- How to define and train the SRNN
- How to generate synthetic data
- Example workflow with the utilities provided in `src/`

---

## 📂 Directory Structure

```
stochastic-rnn/
├── LICENSE
├── pyproject.toml
├── README.md
├── src/                                  # Core SRNN model files
├── quickstart_with_synthetic_example/    # Interactive Jupyter notebook demo
└── atmospheric_regimes_modeling/    # Interactive Jupyter notebook demo
```

---

## 📝 TODOs
  
- [ ] Expand this README with additional experiments and results  
- [ ] Add detailed usage instructions for model customization  

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

Also, if you re-use or derive from the code here, please acknowledge this repository.
