# KyMLP-LDPC System

Post-quantum secure communication system combining Kyber encryption with MLP-LDPC error correction.

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@article{Nguyen2024LDPC,
  author    = {Linh Nguyen and Quoc Bao Phan and Tuy Tan Nguyen},
  title     = {Highly Reliable and Secure System with Multi-layer Parallel LDPC and Kyber for 5G Communications},
  journal   = {IEEE Access},
  volume    = {12},
  pages     = {157260--157271},
  year      = {2024},
  month     = {October},
  doi={10.1109/ACCESS.2024.3485875},  
}

```

## Project Structure

```
kymlp-ldpc/
├── main.py                    # Main entry point
├── core/
│   ├── ldpc.py               # LDPC encoding/decoding
│   ├── preprocess.py         # Data preprocessing and encryption
│   └── kyber_wrapper.py      # Kyber encryption wrapper
├── system/
│   └── kymlp_ldpc.py         # Complete system integration
├── BG/
│   ├── R1-1711982_BG1.xlsx  # Base graph 1
│   └── R1-1711982_BG2.xlsx  # Base graph 2
├── data/                     # Input files (images, signals)
└── output/                   # Output files
```

## Requirements

```bash
pip install numpy pandas pillow kyber-py openpyxl
```

## Quick Start

### 1. Run Complete System
```bash
python main.py
```

Select from menu:
- **Option 1**: Process image/signal files end-to-end
- **Option 2**: Run BER vs SNR simulation
- **Option 3**: Compare encryption overhead

### 2. Process a File
```python
from main import setup_system

system = setup_system(K=256, R=2/3)
stats = system.process_file_end_to_end(
    'data/test.png',
    'output/decoded.png',
    EbNo_dB=3.0,
    iteration=1
)
```

### 3. Run BER Simulation
```python
import numpy as np
from main import setup_system

system = setup_system()
msg_list = [np.random.randint(0, 2, 256) for _ in range(100)]
snr_range = np.linspace(1.0, 3.0, 10)
snr_list, ber_list = system.run_ber_simulation(msg_list, snr_range)
```

## System Flow

```
Input → Encryption → Bit Transform → Segmentation → LDPC Encode 
  → BPSK + AWGN Channel → LDPC Decode → De-bit Transform 
  → Decryption → Output
```

## Parameters

- **K**: Information block size (default: 256 bits)
- **R**: Code rate (default: 2/3)
- **Security Level**: Kyber-512, 768, or 1024
- **EbNo_dB**: Channel SNR in dB
- **Iteration**: LDPC decoding iterations

## Module Usage

### Kyber Wrapper
```python
from core.kyber_wrapper import KyberWrapper

kyber = KyberWrapper(512)
kyber.generate_keys()
ciphertext, encrypted = kyber.encrypt_block(msg)
decrypted = kyber.decrypt_block(ciphertext, encrypted)
```

### Preprocessor
```python
from core.preprocess import EncryptedPreprocessor

preprocessor = EncryptedPreprocessor(kyber, K=256)
blocks, ciphertexts, enc_blocks, metadata = \
    preprocessor.process_with_encryption('input.png')
```

### LDPC Encoder/Decoder
```python
from core.ldpc import LDPC_encoding, process_block

# Encoding
encoded = LDPC_encoding(msg, kb, Z_c, A_sparse, B_sparse, C_sparse, D_sparse)

# Decoding (serial or parallel)
# Serial mode - single process
bit_errors, block_error, received = process_block(
    iteration, std_noise, mbRM, K, kb, Z_c,
    A_sparse, B_sparse, C_sparse, D_sparse,
    layer_params_dict, msg
)
```

### LDPC Processing Modes

The LDPC implementation supports three execution modes:

**1. Serial Processing** (Default)
- Single-threaded execution
- Best for small datasets or debugging
- Lower memory usage
- Saves decoded output to files

**2. Parallel Version 1** 
- Multi-process execution using ProcessPoolExecutor
- Processes multiple blocks simultaneously
- Best for large batch simulations
- Higher memory usage but faster throughput

**3. Parallel Version 2**
- Combines multi-process (blocks) + multi-thread (layers)
- Uses ThreadPoolExecutor for layer processing
- Best for complex decoding scenarios
- Maximum parallelization

Run from command line:
```bash
# In core/ldpc.py, select mode when prompted:
# 1 = Serial
# 2 = Parallel v1 (multi-process)
# 3 = Parallel v2 (multi-process + multi-thread)
# 4 = Run all tests for comparison
```

### LDPC Base Graphs

Supports 5G NR LDPC base graphs:
- **BG1**: For higher code rates (R > 2/3), larger blocks (K > 3840)
- **BG2**: For lower code rates (R <= 2/3), smaller blocks (K <= 3840)

Base graphs use circulant permutation matrices (CPM) with lifting sizes determined by the `Z_c` parameter.

## Notes

- Place base graph files in `BG/` folder
- Input files go in `data/` folder
- Results saved to `output/` folder

- Supports images (PNG, JPG) and signals (TXT, CSV, binary)
