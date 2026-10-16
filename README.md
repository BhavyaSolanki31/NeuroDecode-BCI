# NeuroDecode-BCI 🧠

### Deep Learning-Based Brain–Computer Interface Using EEG Signals

## About the Project

NeuroDecode-BCI is our final-year project focused on building a Brain–Computer Interface (BCI) that can recognize imagined movements from EEG (Electroencephalography) signals.

The basic idea is to understand whether a computer can identify which movement a person is imagining, such as moving their left hand, right hand, or feet, without the person actually moving. Since EEG signals are often noisy and differ from one person to another, making accurate predictions is a challenging task.

Through this project, we aim to explore how deep learning can help interpret these signals and use the predicted movements to control a virtual object or cursor.

## What We Plan to Do

- Work with a publicly available EEG motor imagery dataset.
- Process EEG signals to make them suitable for machine learning.
- Build and compare CNN, LSTM/GRU, and CNN-LSTM models.
- Evaluate how well the models classify different imagined movements.
- Test whether a trained model can generalize to people whose data it has not seen during training.
- Use prediction confidence to avoid acting on uncertain outputs.
- Develop a simple virtual BCI prototype to demonstrate the results.

## Technologies We’re Using

- **Python** — main programming language
- **MNE-Python** — EEG data processing and analysis
- **NumPy and SciPy** — numerical and signal-processing operations
- **scikit-learn** — baseline models and evaluation metrics
- **TensorFlow/Keras** — deep learning models
- **VS Code** — development environment

## How the Project Will Work

Our workflow will begin with understanding the dataset and preparing the EEG signals. We will then develop a baseline model, train different deep learning architectures, and compare their performance using appropriate evaluation metrics.

We will also investigate performance on unseen subjects rather than relying only on results from familiar data. Finally, we plan to connect reliable predictions to a virtual control interface.

## Current Status

The repository source files are visible in VS Code and the Git command-not-found issue appears to have been resolved
after installing Git. The data folder and expected EEG data outputs were not found in the project during the latest checks.
Therefore, the next immediate task is to read the project instructions and locate the data before running preprocessing.
EEGNet development and model comparison remain planned work, not completed results.
The EEG dataset is still to be located and verified locally before running the preprocessing pipeline. After confirming the dataset and preprocessing outputs, the planned next development stage is to implement the EEGNet deep-learning model and compare its performance with the existing CSP-LDA baseline.

## Research Approach

We aim to keep our experiments reproducible by documenting the dataset, preprocessing steps, model configurations, and evaluation results. We will report measured results rather than assume that one model performs better than another.

## Project Goal

Our goal is to develop and evaluate a working EEG-based motor imagery BCI prototype that demonstrates how imagined movements can be translated into virtual commands.

This project is being developed as part of our final-year engineering work, with a focus on learning, experimentation, and practical implementation.
