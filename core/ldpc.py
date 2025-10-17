import sys
sys.path.append('.')
import numpy as np
from itertools import chain
import pandas as pd
import concurrent.futures
import math
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
import gc
import os
from decimal import Decimal, getcontext

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

K = 256
R = 2/3  # Change code rate with 1/3, 2/5, 1/2, 2/3
ITERATION = 10

# SNR range for simulation
EbNodB = np.linspace(0.0, 7.0, 20)

# ============================================================================
# LIFTING SIZE INITIALIZATION
# ============================================================================

a_array = [2, 3, 5, 7, 9, 11, 13, 15]
lifting_size_matrix = np.zeros((len(a_array), 8), dtype=int)
sets = [set() for _ in range(len(a_array))]

for i, a in enumerate(a_array):
    if a <= 3:
        j_range = range(0, 8)
    elif 7 <= a <= 11:
        j_range = range(0, 6)
    elif a >= 13:
        j_range = range(0, 5)
    else:
        j_range = range(0, 7)
    
    for j in j_range:
        lifting_size_matrix[i][j] = a * (2 ** j)
        sets[i].add(lifting_size_matrix[i][j])

df_Z = pd.DataFrame(lifting_size_matrix, columns=range(8), dtype=int)

# ============================================================================
# BASE GRAPH SELECTION
# ============================================================================

def determine_B_and_kb(K, R):
    """Determine base graph and kb based on K and R."""
    if K > 3840:
        B = 1
    elif K <= 308:
        B = 2
    else:
        B = 1 if R > (2/3) else 2
    
    kb = 22 if B == 1 else 10
    return B, kb

base_graph, kb = determine_B_and_kb(K, R)

# ============================================================================
# MATRIX INITIALIZATION
# ============================================================================

# Determine the base matrix expansion factor Zc
Z_c = int(df_Z[df_Z * kb >= K].min().min())
nbRM = math.ceil(kb / R) + 2
n = nbRM * Z_c
mbRM = nbRM - kb

# Find which set contains Z_c
index_of_set_with_Z_c = None
for i, s in enumerate(sets):
    if Z_c in s:
        index_of_set_with_Z_c = i + 1
        break

# Load appropriate base graph
if base_graph == 2:
    nb = 52
    mb = 42
    bg_file = 'BG/R1-1711982_BG2.xlsx'
else:
    nb = 68
    mb = 46
    bg_file = 'BG/R1-1711982_BG1.xlsx'

bg_data = pd.ExcelFile(bg_file)
sheet_names = bg_data.sheet_names
prefix = 'Set ' + str(index_of_set_with_Z_c)

for sheet in sheet_names:
    if sheet.startswith(prefix):
        base_graph_df = pd.read_excel(bg_file, sheet_name=sheet, header=None)
        break

# Create parity check matrix P
P = np.array([
    [-1 if base_graph_df.iloc[i, j] == -1 else int(base_graph_df.iloc[i, j]) % Z_c 
     for j in range(nbRM)] 
    for i in range(mbRM)
])

print("Matrix P shape:", P.shape)
print("kb:", kb, "Z_c:", Z_c)

# ============================================================================
# MATRIX OPERATIONS
# ============================================================================

def sub_matrix(input_matrix, start_row_idx, end_row_idx, start_col_idx, end_col_idx):
    """Extract submatrix."""
    return input_matrix[start_row_idx:end_row_idx, start_col_idx:end_col_idx]

def binary_CPM(element, Z):
    """Generate binary circulant permutation matrix."""
    if element == -1:
        return np.zeros((Z, Z), dtype=int)
    
    dispersed_matrix = np.eye(Z, dtype=int)
    
    if element >= 1:
        row_indices, col_indices = np.where(dispersed_matrix == 1)
        new_col_indices = (col_indices + element) % Z
        dispersed_matrix = np.zeros((Z, Z), dtype=int)
        dispersed_matrix[row_indices, new_col_indices] = 1
    
    return dispersed_matrix

def create_sparse_matrix(input_matrix, size):
    """Create sparse parity check matrix from compressed representation."""
    cols, rows = len(input_matrix), len(input_matrix[0])
    cpm_matrix = np.zeros((cols * size, rows * size), dtype=int)
    
    for i in range(cols):
        for j in range(rows):
            dispersed_binary_CPM = binary_CPM(input_matrix[i][j], size)
            cpm_matrix[i * size:(i + 1) * size, j * size:(j + 1) * size] = dispersed_binary_CPM
    
    return cpm_matrix

def H_submatrices(P, Z_c):
    """Create H matrix and its submatrices."""
    g = 4
    mb, nb = P.shape
    
    # Define submatrices
    A_matrix = sub_matrix(P, 0, g, 0, kb)
    B_matrix = sub_matrix(P, 0, g, kb, kb + g)
    C_matrix = sub_matrix(P, g, mb, 0, kb)
    D_matrix = sub_matrix(P, g, mb, kb, kb + g)
    I_matrix = sub_matrix(P, g, mb, kb + g, nb)
    O_matrix = sub_matrix(P, 0, g, kb + g, nb)
    
    # Create sparse matrices
    A_sparse = create_sparse_matrix(A_matrix, Z_c)
    B_sparse = create_sparse_matrix(B_matrix, Z_c)
    C_sparse = create_sparse_matrix(C_matrix, Z_c)
    D_sparse = create_sparse_matrix(D_matrix, Z_c)
    I_sparse = create_sparse_matrix(I_matrix, Z_c)
    O_sparse = create_sparse_matrix(O_matrix, Z_c)
    
    H = np.block([[A_sparse, B_sparse, O_sparse], [C_sparse, D_sparse, I_sparse]])
    
    return H, A_sparse, B_sparse, O_sparse, C_sparse, D_sparse, I_sparse

H, A_sparse, B_sparse, O_sparse, C_sparse, D_sparse, I_sparse = H_submatrices(P, Z_c)
L = mbRM
L_matrices = np.array_split(H, L)

# ============================================================================
# ENCODING
# ============================================================================

def LDPC_encoding(msg, kb, Z_c, A_sparse, B_sparse, C_sparse, D_sparse):
    """Encode message using LDPC code."""
    s = np.zeros(kb * Z_c)
    if kb * Z_c > len(msg):
        s[:K] = msg
    elif kb * Z_c == len(msg):
        s = msg
    
    B_inv = np.linalg.inv(B_sparse).astype(int)
    p1 = np.dot(B_inv, np.dot(A_sparse, s)) % 2
    p2 = (C_sparse @ s + D_sparse @ p1) % 2
    
    encoded_bits = np.concatenate((s, p1, p2))
    return encoded_bits

# ============================================================================
# DECODING UTILITIES
# ============================================================================

def calculate_function(Pn):
    """Hard decision function."""
    return (1 - np.sign(Pn)) // 2

def create_check_node_connection_sets(H):
    """Create dictionary of check node connections."""
    dictionary = {}
    for i, row in enumerate(H):
        dictionary[i + 1] = [j + 1 for j, value in enumerate(row) if value == 1]
    return dictionary

def get_not_minus_one_indices(column_values):
    """Get indices of non -1 values."""
    not_minus_one_indices = np.atleast_1d(column_values != -1).nonzero()[0].tolist()
    unique_value_indices = []
    value_to_index = {}
    
    for idx in not_minus_one_indices:
        value = column_values[idx]
        if value not in value_to_index:
            value_to_index[value] = idx
            unique_value_indices.append(idx + 1)
    
    return unique_value_indices

def init_node(mbRM):
    """Initialize node structure."""
    node = {}
    for layer_idx in range(mbRM):
        node[layer_idx + 1] = {"node_data": {}}
    return node

def init_check_node_value(layer_idx, layer, check_node_updates_json):
    """Initialize check node values."""
    connection_set = create_check_node_connection_sets(layer)
    for m, val in connection_set.items():
        for n in val:
            check_node_updates_json[layer_idx]["node_data"].update({(m, n): 0 for n in val})
    return check_node_updates_json[layer_idx]["node_data"]

# ============================================================================
# LAYER PROCESSING
# ============================================================================

# Build multi-layer set
multi_layers_set = {}
i = 0
col_idx = 0

while col_idx < P.shape[1]:
    layer_idx_set = set(range(1, P.shape[0] + 1)) - set(list(chain.from_iterable(multi_layers_set.values())))
    while layer_idx_set:
        layer_value = np.array([P[idx - 1, col_idx] for idx in layer_idx_set])
        new_layer_idx = get_not_minus_one_indices(layer_value)
        if not new_layer_idx:
            col_idx += 1
            break
        multi_layers_set[i + 1] = [list(layer_idx_set)[idx - 1] for idx in new_layer_idx]
        layer_idx_set -= set(multi_layers_set[i + 1])
        i += 1
    else:
        col_idx += 1

# Prepare layer parameters
layer_params_dict = {}
for idx, layer_set in multi_layers_set.items():
    layer_params = [(val, L_matrices[val - 1], Z_c) for val in layer_set]
    layer_params_dict[idx] = layer_params

def process_layer(layer_idx, layer, Z_c, check_node_updates_json, variable_node_updates_json, LLR_dict):
    """Process a single layer in the decoding algorithm."""
    connection_set = create_check_node_connection_sets(layer)
    check_node_updates_json[layer_idx]["node_data"] = init_check_node_value(
        layer_idx, layer, check_node_updates_json
    )
    
    # Update variable nodes
    for m, val in connection_set.items():
        for n in val:
            variable_node_updates_json[layer_idx]["node_data"].update({
                (m, n): LLR_dict[n] - check_node_updates_json[layer_idx]["node_data"][(m, n)] 
                for n in val
            })
    
    # Update check nodes and posteriori info
    for mx, value in connection_set.items():
        for n in value:
            n_prime_set = [n_prime for n_prime in value if n_prime != n]
            filtered_row = [variable_node_updates_json[layer_idx]["node_data"][mx, n_prime] 
                          for n_prime in n_prime_set]
            min_value = np.min(np.abs(filtered_row))
            sign = [np.sign(value) if np.sign(value) != 0 else 1 for value in filtered_row]
            signs_product = np.prod(sign)
            check_node_updates_json[layer_idx]["node_data"][mx, n] = min_value * signs_product
            
            # Posteriori update
            LLR_dict[n] = (variable_node_updates_json[layer_idx]["node_data"][mx, n] + 
                          check_node_updates_json[layer_idx]["node_data"][mx, n])
    
    return LLR_dict

# ============================================================================
# NOISE GENERATION
# ============================================================================

sigma_square_list = np.zeros(len(EbNodB))
for i in range(len(EbNodB)):
    EbNo = 10 ** (EbNodB[i] / 10)
    sigma_square_list[i] = 1 / (2 * R * EbNo)

std_noise_list = []
for sigma_square in sigma_square_list:
    noise = np.random.randn(H.shape[1])
    std_noise = noise * sigma_square
    std_noise_list.append(std_noise)

# ============================================================================
# BLOCK PROCESSING
# ============================================================================

def process_block(iteration, std_noise, mbRM, K, kb, Z_c, A_sparse, B_sparse, 
                 C_sparse, D_sparse, layer_params_dict, msg):
    """Process a single block (with received bits output)."""
    encoded_bits = LDPC_encoding(msg, kb, Z_c, A_sparse, B_sparse, C_sparse, D_sparse)
    symbols = 1 - 2 * encoded_bits
    LLR_received = symbols + std_noise
    
    # Apply puncturing and shortening
    LLR_received[:2 * Z_c] = 0
    if K < kb * Z_c:
        LLR_received[K:kb * Z_c] = 10000000
    
    LLR_dict = {index + 1: value for index, value in enumerate(LLR_received)}
    variable_node_updates_json = init_node(mbRM)
    check_node_updates_json = init_node(mbRM)
    
    for _ in range(iteration):
        for idx in sorted(layer_params_dict.keys()):
            params = layer_params_dict[idx]
            results = []
            for param in params:
                LLR_dict_partial = process_layer(*param, check_node_updates_json, 
                                                variable_node_updates_json, LLR_dict)
                results.append(LLR_dict_partial)
            for LLR_dict_partial in results:
                LLR_dict.update(LLR_dict_partial)
    
    received_bits = np.zeros(len(LLR_dict), dtype=int)
    for key, value in LLR_dict.items():
        received_bits[key - 1] = calculate_function(value)
    
    bit_errors = np.sum(msg != received_bits[:K])
    block_error = int(bit_errors > 0)
    
    return bit_errors, block_error, received_bits[:K]

def one_block_process(iteration, std_noise, mbRM, K, kb, Z_c, A_sparse, B_sparse, 
                     C_sparse, D_sparse, layer_params_dict, msg):
    """Process a single block with threading (without received bits output)."""
    encoded_bits = LDPC_encoding(msg, kb, Z_c, A_sparse, B_sparse, C_sparse, D_sparse)
    symbols = 1 - 2 * encoded_bits
    LLR_received = symbols + std_noise
    
    # Apply puncturing and shortening
    LLR_received[:2 * Z_c] = 0
    if K < kb * Z_c:
        LLR_received[K:kb * Z_c] = 10000000
    
    LLR_dict = {index + 1: value for index, value in enumerate(LLR_received)}
    variable_node_updates_json = init_node(mbRM)
    check_node_updates_json = init_node(mbRM)
    
    for _ in range(iteration):
        for idx in sorted(layer_params_dict.keys()):
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_layer, *params, check_node_updates_json, 
                                          variable_node_updates_json, LLR_dict) 
                          for params in layer_params_dict[idx]]
                for future in futures:
                    LLR_dict_partial = future.result()
                    for key, value in LLR_dict_partial.items():
                        LLR_dict[key] = value
    
    received_bits = np.zeros(len(LLR_dict), dtype=int)
    for key, value in LLR_dict.items():
        received_bits[key - 1] = calculate_function(value)
    
    bit_errors = np.sum(msg != received_bits[:K])
    block_error = int(bit_errors > 0)
    
    return bit_errors, block_error

# ============================================================================
# INPUT LOADING
# ============================================================================

def load_messages_from_file(file_path, K):
    """Load messages from a text file."""
    msg_list = []
    with open(file_path, 'r') as file:
        binary_numbers = ''
        for line in file:
            binary_numbers += line.strip()
        
        for i in range(0, len(binary_numbers), K):
            msg_list.append([int(char) for char in binary_numbers[i:i + K]])
    
    return msg_list

def generate_random_messages(num_blocks, K):
    """Generate random binary messages."""
    return [np.random.randint(2, size=K).tolist() for _ in range(num_blocks)]

# ============================================================================
# SIMULATION MODES
# ============================================================================

def run_serial(msg_list, save_output=True):
    """Run serial computation."""
    ber_snr1 = []
    print("\n" + "="*60)
    print("Starting Serial Computation")
    print("="*60)
    serial_start_time = time.time()
    stop_serial = False
    
    getcontext().prec = 15
    K_d = Decimal(K)
    len_msg_list = Decimal(len(msg_list))
    
    for idx, std_noise in enumerate(std_noise_list):
        if stop_serial:
            break
        
        eb_no_dB = EbNodB[idx]
        print(f"Processing SNR {idx + 1}/{len(std_noise_list)}: Eb/N0 = {eb_no_dB:.2f} dB")
        Nblkerrs_total = 0
        Nerr_total = 0
        
        if save_output:
            os.makedirs("output", exist_ok=True)
            output_file = f"output/received_bits_{eb_no_dB:.1f}dB.txt"
            
            with open(output_file, 'w') as file:
                for msg in msg_list:
                    if stop_serial:
                        break
                    
                    Nerr, Blkerr, received = process_block(
                        ITERATION, std_noise, mbRM, K, kb, Z_c,
                        A_sparse, B_sparse, C_sparse, D_sparse,
                        layer_params_dict, msg
                    )
                    
                    Nerr_total += Nerr
                    Nblkerrs_total += Blkerr
                    
                    if Nerr_total == 0 and Nblkerrs_total == 0:
                        stop_serial = True
                        break
                    
                    received_str = ''.join(map(str, received))
                    file.write(received_str)
        else:
            for msg in msg_list:
                if stop_serial:
                    break
                
                Nerr, Blkerr, received = process_block(
                    ITERATION, std_noise, mbRM, K, kb, Z_c,
                    A_sparse, B_sparse, C_sparse, D_sparse,
                    layer_params_dict, msg
                )
                
                Nerr_total += Nerr
                Nblkerrs_total += Blkerr
                
                if Nerr_total == 0 and Nblkerrs_total == 0:
                    stop_serial = True
                    break
        
        Nerr_total_d = Decimal(int(Nerr_total))
        ber = Nerr_total_d / (K_d * len_msg_list)
        ber_snr1.append(ber)
        print(f"  Bit errors: {Nerr_total}, Block errors: {Nblkerrs_total}, BER: {ber:.15f}")
    
    ber_snr1_sorted = sorted(ber_snr1, reverse=True)
    formatted_bers = [f"{ber:.15f}" for ber in ber_snr1_sorted]
    formatted_output = ", ".join(formatted_bers)
    print(f"\nBERs (sorted): [{formatted_output}]")
    
    serial_end_time = time.time()
    serial_duration = serial_end_time - serial_start_time
    print(f"\nSerial computation time: {serial_duration:.2f} seconds")
    print("="*60 + "\n")

def run_parallel_v1(msg_list):
    """Run parallel computation version 1."""
    ber_snr_parallel = []
    print("\n" + "="*60)
    print("Starting Parallel Computation Version 1")
    print("="*60)
    parallel_start_time1 = time.time()
    stop_processing = False
    batch_size = len(msg_list)
    
    for idx, std_noise in enumerate(std_noise_list):
        if stop_processing:
            break
        
        eb_no_dB = EbNodB[idx]
        print(f"Processing SNR {idx + 1}/{len(std_noise_list)}: Eb/N0 = {eb_no_dB:.2f} dB")
        Nblkerrs_total1 = 0
        Nerr_total1 = 0
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for i in range(0, len(msg_list), batch_size):
                if stop_processing:
                    break
                
                batch_msgs = msg_list[i:i + batch_size]
                futures.extend([
                    executor.submit(process_block, ITERATION, std_noise, mbRM, K, kb, Z_c, 
                                  A_sparse, B_sparse, C_sparse, D_sparse, layer_params_dict, msg) 
                    for msg in batch_msgs
                ])
            
            for future in concurrent.futures.as_completed(futures):
                Nerr1, Blkerr1, _ = future.result()
                Nerr_total1 += Nerr1
                Nblkerrs_total1 += Blkerr1
                if Nerr_total1 == 0 and Nblkerrs_total1 == 0:
                    stop_processing = True
                    break
        
        ber1 = Nerr_total1 / (K * len(msg_list))
        ber_snr_parallel.append(ber1)
        print(f"  Bit errors: {Nerr_total1}, Block errors: {Nblkerrs_total1}, BER: {ber1:.15f}")
    
    ber_snr_parallel = sorted(ber_snr_parallel, reverse=True)
    print(f"\nBERs (sorted): {ber_snr_parallel}")
    
    parallel_end_time1 = time.time()
    parallel_duration1 = parallel_end_time1 - parallel_start_time1
    print(f"\nParallel computation time: {parallel_duration1:.2f} seconds")
    print("="*60 + "\n")

def run_parallel_v2(msg_list):
    """Run parallel computation version 2."""
    ber_snr2 = []
    print("\n" + "="*60)
    print("Starting Parallel Computation Version 2")
    print("="*60)
    parallel_start_time2 = time.time()
    stop_processing2 = False
    batch_size = len(msg_list)
    
    for idx, std_noise in enumerate(std_noise_list):
        if stop_processing2:
            break
        
        eb_no_dB = EbNodB[idx]
        print(f"Processing SNR {idx + 1}/{len(std_noise_list)}: Eb/N0 = {eb_no_dB:.2f} dB")
        Nblkerrs_total2 = 0
        Nerr_total2 = 0
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for i in range(0, len(msg_list), batch_size):
                if stop_processing2:
                    break
                batch_msgs = msg_list[i:i + batch_size]
                futures.extend([
                    executor.submit(one_block_process, ITERATION, std_noise, mbRM, K, kb, Z_c, 
                                  A_sparse, B_sparse, C_sparse, D_sparse, layer_params_dict, msg) 
                    for msg in batch_msgs
                ])
            
            for future in concurrent.futures.as_completed(futures):
                Nerr2, Blkerr2 = future.result()
                Nerr_total2 += Nerr2
                Nblkerrs_total2 += Blkerr2
                if Nerr_total2 == 0 and Nblkerrs_total2 == 0:
                    stop_processing2 = True
                    break
        
        ber2 = Nerr_total2 / (K * len(msg_list))
        ber_snr2.append(ber2)
        print(f"  Bit errors: {Nerr_total2}, Block errors: {Nblkerrs_total2}, BER: {ber2:.15f}")
    
    ber_snr2 = sorted(ber_snr2, reverse=True)
    print(f"\nBERs (sorted): {ber_snr2}")
    
    parallel_end_time2 = time.time()
    parallel_duration2 = parallel_end_time2 - parallel_start_time2
    print(f"\nParallel computation time: {parallel_duration2:.2f} seconds")
    print("="*60 + "\n")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("LDPC ENCODER/DECODER SIMULATION")
    print("="*60)
    print(f"Configuration: K={K}, R={R}, Iterations={ITERATION}")
    print("="*60 + "\n")
    
    # Input selection
    print("Select input type:")
    print("1. Load from text file")
    print("2. Generate random messages")
    input_choice = int(input("Enter your choice (1/2): "))
    
    if input_choice == 1:
        file_path = input("Enter the file path: ")
        msg_list = load_messages_from_file(file_path, K)
        print(f"Loaded {len(msg_list)} blocks from file")
    elif input_choice == 2:
        num_blocks = int(input("Enter number of random blocks to generate: "))
        msg_list = generate_random_messages(num_blocks, K)
        print(f"Generated {len(msg_list)} random blocks")
    else:
        print("Invalid choice. Using default: 5 random blocks")
        msg_list = generate_random_messages(5, K)
    
    print(f"\nTotal blocks to process: {len(msg_list)}")
    print("="*60 + "\n")
    
    # Execution mode selection
    print("Select execution mode:")
    print("1. Serial computation")
    print("2. Parallel computation version 1")
    print("3. Parallel computation version 2")
    print("4. Run all tests")
    
    option = int(input("Enter your choice (1/2/3/4): "))
    
    if option == 1:
        run_serial(msg_list)
    elif option == 2:
        run_parallel_v1(msg_list)
    elif option == 3:
        run_parallel_v2(msg_list)
    elif option == 4:
        run_serial(msg_list)
        gc.collect()
        time.sleep(1)
        run_parallel_v1(msg_list)
        gc.collect()
        time.sleep(1)
        run_parallel_v2(msg_list)
    else:
        print("Invalid option. Please enter 1, 2, 3, or 4.")