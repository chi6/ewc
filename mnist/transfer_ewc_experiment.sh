#!/bin/bash
# Shell script for running transfer on MNIST experiemnt

# 1. Pre-transfer: Train entire network on MNIST 0-4
python baseline_0-4.py

# 2. Transfer: Retrain/test network on MNIST 5-9
python transfer_ewc.py

# 3. Post-transfer: Test retrained network on MNIST 0-4
python post_transfer_ewc.py
