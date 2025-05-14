# 🌱 Greensight - A Climate Change Modeler

A hybrid time series modeling project exploring **statistical** and **deep learning** methods to forecast environmental variables such as temperature using the [Jena Climate Dataset](https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip). The project compares **SARIMA** (Seasonal AutoRegressive Integrated Moving Average) with **LSTM** (Long Short-Term Memory) networks to evaluate performance, interpretability, and computational efficiency.

---

## 🔍 Project Goals

- Predict future climate trends from historical weather data.
- Compare performance between a classical time series model (SARIMA) and a neural model (LSTM).
- Use **JAX** to accelerate numerical computations and reduce evaluation time.
- Evaluate models with metrics like **sMAPE** and **RMSE**, and visualize prediction intervals.

---

## 📊 Dataset

- **Source**: Jena Climate Dataset (2009–2016)  
- **Link**: [jena_climate_2009_2016.csv.zip](https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip)  
- **Variables**: Temperature, pressure, humidity, wind speed, etc.  
- **Frequency**: Hourly → Aggregated to **daily mean** for modeling.

---

## 🧠 Models Implemented

### 1. **SARIMA (Statistical)**
- Built using `statsmodels` and accelerated with `jax.numpy`.
- Automatically selects optimal lag order based on autocorrelation analysis.
- Captures seasonality using SARIMA's `(p, d, q)(P, D, Q, s)` hyperparameters.

### 2. **LSTM (Neural Network)**
- Implemented in TensorFlow/Keras.
- Sequence-to-one architecture with single-step prediction.
- Trained on sliding windows of past timesteps.
- Takes longer to train but is flexible for multi-variate and non-linear patterns.

---

## 📈 Evaluation Metrics

| Model  | sMAPE ↓ | RMSE ↓ | Time (Approx.) |
|--------|---------|--------|----------------|
| SARIMA | **38.67** | 2.29   | ~20 seconds     |
| LSTM   | 61.49    | **0.35** | ~20 minutes     |

- **sMAPE** (Symmetric Mean Absolute Percentage Error) – scale-independent, interpretable.
- **RMSE** (Root Mean Squared Error) – sensitive to large errors.

---

## 📦 Tech Stack

- **Python 3.11**
- `statsmodels`, `jax`, `jax.numpy`
- `tensorflow`, `keras`, `matplotlib`, `pandas`, `numpy`, `scikit-learn`

---

## 📌 Key Learnings

- SARIMA models are highly efficient and interpretable for seasonal data, but limited in modeling complex nonlinearities.
- LSTMs are flexible but computationally expensive and require careful tuning.
- JAX significantly boosts runtime performance for numerical-heavy forecasting tasks.
- Evaluation metrics and visualizations are essential to draw real insights from model performance.

---

## 🚀 Future Work

- Multi-step forecasting with LSTM and Transformer-based models.
- Expand to multivariate modeling using all available features.
- Deploy as a real-time forecasting dashboard (Streamlit/Gradio).
- Integrate hyperparameter tuning frameworks (Optuna/Hyperopt).
