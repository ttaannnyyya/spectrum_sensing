# 📡 **Deep Learning–Based Spectrum Sensing (DetectNet)**

## **Overview**
This project focuses on detecting the presence of wireless signals in a **cognitive radio system**, where secondary users can access the spectrum only when primary users are inactive. The goal is to reliably detect signals even under **low Signal-to-Noise Ratio (SNR)** conditions.

## **Dataset**
Wireless signals are generated using **GNU Radio** with multiple modulation schemes (e.g., QPSK, QAM). Realistic channel noise is added, and the signals are sliced into fixed-length **I/Q time-series samples** for training.

## **Model**
The core model, **DetectNet**, uses a **CNN + LSTM architecture**:
- **CNN layers** extract local patterns from I/Q signals.
- **LSTM layers** capture temporal dependencies across time.

## **Evaluation**
Model performance is evaluated using **Probability of Detection (Pd)** and **Probability of False Alarm (Pf)**. DetectNet achieves a **~5 dB SNR improvement** over traditional energy detectors while maintaining **>90% detection probability**.

## **Impact**
This approach enables efficient and safe spectrum utilization by secondary users without interfering with primary users.
