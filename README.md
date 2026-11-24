# Detecting Group Temporal Key (GTK) Abuse in 802.11 Networks with Isolation Forest

MSc Cyber Security dissertation project, Northumbria University (Distinction; dissertation graded A).

In a WPA2/WPA3 network, every connected client shares the Group Temporal Key (GTK). That means an insider can abuse it to inject forged broadcast or multicast traffic that looks legitimate at the Wi-Fi layer. This project tests whether **unsupervised anomaly detection** can catch these attacks using only features from captured traffic.

## Results

I trained and compared two Isolation Forest models:

- **Layer 2 model:** 802.11 frame features only
- **Cross-layer model:** 802.11 features combined with IP and ARP features

The **cross-layer model performed best, with 92.3% recall and a 72.8% F1 score**. It uses features that measure inconsistencies between the Wi-Fi layer and the network layer, such as broadcast MAC frames carrying non-broadcast IP traffic. A simple rule-based baseline (`rule_based.py`) is included for comparison.

## How it works

1. **Capture:** a WLAN testbed (access point, attacker machine, USB Wi-Fi adapters and several client devices) captured **40,000+ packets** of normal and malicious traffic in Wireshark.
2. **Feature extraction** (`extract_features.py`): Scapy turns raw captures into fixed time windows with features such as frame-type counts, broadcast/multicast/unicast MAC ratios, EAPOL and protected-frame counts, packet rate, and cross-layer inconsistency counts.
3. **Combine sessions** (`combine_normal_capture.py`): merges normal traffic from separate capture sessions into one dataset.
4. **Prepare data** (`prepare_data.py`): selects features and builds the Layer 2 and cross-layer feature sets.
5. **Train** (`train_model.py`): trains two scikit-learn Isolation Forest models on **normal traffic only**, so no attack labels are needed for training.
6. **Evaluate** (`evaluate_model.py`): scores both models against labelled attack windows and reports precision, recall and F1.

## Repository structure

```
├── extract_features.py         # pcap → windowed feature CSV (Scapy)
├── combine_normal_capture.py   # merge normal capture sessions
├── prepare_data.py             # feature selection and dataset preparation
├── train_model.py              # train Layer 2 and cross-layer Isolation Forests
├── evaluate_model.py           # evaluation metrics
├── rule_based.py               # rule-based baseline
├── prepared_data/              # windowed, aggregated features (no raw packets or MAC addresses)
└── models/                     # trained models and training metadata
```

## Tech stack

Python · Scapy · pandas · scikit-learn · Wireshark

## Note on data

Only aggregated, windowed features are published. Raw packet captures aren't included.
