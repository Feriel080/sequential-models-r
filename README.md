# sequential-models-r

![R Version](https://img.shields.io/badge/R-4.5.2-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

Comparison of the performance of sequential models using **R**

## _NOTICE: THIS WORK IS FOR LEARNING PURPOSE_

## Used datasets

* **Time series dataset: [Jena Climate](https://www.kaggle.com/datasets/mnassrib/jena-climate)**:
  * Target to forecast: **Pressure** (noted as p in the dataset)
  * Lags: 1, 3, 6, 18
  * Rolling mean: 6, 36
  * The lags and rolling mean were combined to give better results.
  * For multivariate, only _p (mbar)_, _T (degC)_, _H2OC (mmol/mol)_, _wv (m/s)_ features were used.

**Text dataset:  [AG News Classification Dataset](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)**

## Installation

Clone the repository:

```bash
git clone <repo url>
```

Install dependencies using R terminal:

```bash
renv::restore()

# Run using this line
source("file_name.r")
```

## Project Structure

| Directory / File                                                                                                                                                                                                    | Description                                                                                                                                                                                                                                                                           |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [text/AG News](https://github.com/Feriel080/sequential-models-r/tree/main/text/AG%20News)                                                                                                                              | AG News cleaned, train and test datasets                                                                                                                                                                                                                                              |
| [text/data_analysis.ipynb](https://github.com/Feriel080/sequential-models-r/blob/main/text/data_analysis.ipynb)                                                                                                        | AG News dataset analysis                                                                                                                                                                                                                                                              |
| [time_series/datasets](https://github.com/Feriel080/sequential-models-r/tree/main/time_series/datasets)                                                                                                                | Jena Climate dataset, along with the lags                                                                                                                                                                                                                                             |
| ([text](https://github.com/Feriel080/sequential-models-r/tree/main/text/results)/[time_series](https://github.com/Feriel080/sequential-models-r/tree/main/time_series/results))/results/                                  | Experiment results of RNN, BRNN, LSTM, GRU, Transformer for univariate and multivariate forecasting. It contains the following:_model checkpoint_, _learning curve plot_, _predicted vs real values plot_, _train and val losses history_, and _full results of each model_ |
| ([text](https://github.com/Feriel080/sequential-models-r/tree/main/text/results/comparison)/[time_series](https://github.com/Feriel080/sequential-models-r/tree/main/time_series/results/comparison))/results/comparison/ | Contains comparison plots:_Best epoch of each model_, _best validation loss_, _r2 score comparison_, _training time comparison_                                                                                                                                               |
| [time_series/data_analysis.ipynb](https://github.com/Feriel080/sequential-models-r/blob/main/time_series/data_analysis.ipynb)                                                                                          | Analysis of Jena Climate dataset                                                                                                                                                                                                                                                      |
| [rnn.r](https://github.com/Feriel080/sequential-models-r/blob/main/rnn.r)                                                                                                                                              | RNN model training and evaluation                                                                                                                                                                                                                                                     |
| [brnn.r](https://github.com/Feriel080/sequential-models-r/blob/main/brnn.r)                                                                                                                                            | Bidirectional RNN model training and evaluation                                                                                                                                                                                                                                       |
| [lstm.r](https://github.com/Feriel080/sequential-models-r/blob/main/lstm.r)                                                                                                                                            | LSTM model training and evaluation                                                                                                                                                                                                                                                    |
| [gru.r](https://github.com/Feriel080/sequential-models-r/blob/main/gru.r)                                                                                                                                              | GRU model training and evaluation                                                                                                                                                                                                                                                     |
| [transformers.r](https://github.com/Feriel080/sequential-models-r/blob/main/transformers.r)                                                                                                                            | Transformers training and evaluation                                                                                                                                                                                                                                                  |
| [comparison.r](https://github.com/Feriel080/sequential-models-r/blob/main/comparison.r)                                                                                                                                | Comparison plots                                                                                                                                                                                                                                                                      |

## Models

* **Classical RNN (Recurrent Neural Networks)**
* **BRNN (Bidirectional Recurrent Neural Networks)**
* **LSTM (Long-short Term Memory)**
* **GRU (Gated Recurrent Unit)**
* **Transformers (used only the encoder part)**

All models shared the following configurations for fair comparison:

### **Configurations for Time Series**:

* Batch size: 64
* Number of epochs: 50
* Hidden size: 64
* Number of layers: 2
* Dropout: 0.2
* Learning rate: 1e-3
* Criterion: MSE Loss
* Optimizer: Adam

**Additional for transformers**:

* Dimension of the model: 64
* Number of heads: 8
* Dimension of feed forward: 128

### **Configurations for Text Classification:**

* Batch size: 128
* Number of epochs: 30
* Hidden size: 128 (for transformers there's no hidden size)
* Max vocabulary size: 25000
* Max Sequence length: 100
* Embedding dimension: 100 (128 for transformers)
* Number of layers: 2
* Dropout: 0.5 (0.3 for transformers)
* Learning rate: 1e-3 (1e-4 for transformers)

**Additional For Transformers:**

* Dimension of the model: 128
* Number of heads: 4
* Dimension of feed forward: 256

## Clarification of Text Classification results

If you go see one of the metrics of the classification results in _text/results/<model_name>/full_results.json_ you'll find a high score. That was due to using the best model obtained from the training which we saved for predicting classes.

## Key Findings

NOT YET
