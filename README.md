# Climate Change Predictor Using Time Series Forecasting

## Project Overview

This repository contains the code and resources for the Climate Change Predictor Using Time Series Forecasting project. This project aims to predict future climate trends, including temperature and rainfall, using time series forecasting techniques. The project leverages a Seasonal ARIMA (SARIMA) model enhanced with the JAX library for improved performance and efficiency.

## Submitted by:

1. Aabir Datta - 19BCE10062
2. Uday Agarwal - 19BAI10147
3. Abdul Raziq Khan - 19BCE10135
4. Parv Bhargava - 19BAI10116
5. Aryan Tandon - 19BAI10148
6. Vimal Tiwari - 19BCG10054
7. Govit Khasare - 19MIM10094
8. Mriganka Shekhar Das - 19BOE10053


## Motivation & Objective

Climate change poses significant threats and challenges globally. By understanding and predicting climate patterns, we can better prepare and mitigate the impacts of extreme weather events. This project aims to:

- Analyze historical climate data.
- Predict future climate conditions using advanced time series forecasting methods.
- Provide actionable insights for mitigating climate change impacts.

## Methodology

### Data Collection
- The dataset used includes global mean monthly temperature and rainfall data spanning over 30 years.
- Data sources include NASA and the World Bank.

### Data Collection
- A SARIMA model is used to fit the time series data for temperature and rainfall predictions.
- The JAX library is incorporated to enhance model performance, significantly reducing training time.

### Results
- The model predicts temperature trends for the next 10 years and rainfall amounts for the next 5 years.
- The enhanced SARIMA model with JAX shows improved training efficiency and prediction accuracy.

## System Architecture

The system architecture involves:
- Data preprocessing and cleaning.
- Time series forecasting using the SARIMA model.
- Performance enhancement using the JAX library.

## Usage

### Prerequisites
- Python 3.8+
- Libraries: pandas, matplotlib, tensorflow, keras, jax

## References

1. Turner, P. (2020). [Time Series Analysis and Climate Change](https://towardsdatascience.com/time-series-analysis-and-climate-change-7bb4371021e).
2. Mudelsee, M. (2019). Trend analysis of climate time series: A review of methods. Earth-Science Reviews, 190, 310-322.
3. Khoa, L. (2019). [Time Series Analysis and Weather Forecast in Python](https://medium.com/@llmkhoa511/time-series-analysis-and-weather-forecast-in-python-e80b664c7f71).
4. Chollet, F. (2019). [Keras Time Series Weather Forecasting](https://keras.io/examples/timeseries/timeseries_weather_forecasting/).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
