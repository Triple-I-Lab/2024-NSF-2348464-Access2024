"""
Main Entry Point for KyMLP-LDPC System
Run experiments and simulations
"""

import numpy as np
import sys
import os

# Add core and system directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'system'))

from core.kyber_wrapper import KyberWrapper
from core.preprocess import EncryptedPreprocessor
from system.kymlp_ldpc import KyMLPLDPCSystem, LDPCEncoderDecoder


def initialize_ldpc(K=256, R=2/3):
    """
    Initialize LDPC encoder/decoder with all required matrices
    This imports from the cleaned ldpc.py
    
    Args:
        K (int): Information block size
        R (float): Code rate
    
    Returns:
        LDPCEncoderDecoder: Initialized LDPC encoder/decoder
    """
    # Import LDPC initialization code
    from core.ldpc import (
        H, A_sparse, B_sparse, C_sparse, D_sparse,
        kb, Z_c, mbRM, layer_params_dict, K as LDPC_K, R as LDPC_R
    )
    
    # Verify parameters match
    if K != LDPC_K or R != LDPC_R:
        print(f"Warning: Requested K={K}, R={R} but LDPC uses K={LDPC_K}, R={LDPC_R}")
        print(f"Using LDPC configuration: K={LDPC_K}, R={LDPC_R}")
        K = LDPC_K
        R = LDPC_R
    
    ldpc = LDPCEncoderDecoder(
        K, R, H, A_sparse, B_sparse, C_sparse, D_sparse,
        kb, Z_c, mbRM, layer_params_dict
    )
    
    print(f"LDPC initialized: K={K}, R={R}, Z_c={Z_c}")
    
    return ldpc


def setup_system(K=256, R=2/3, security_level=512):
    """
    Setup complete KyMLP-LDPC system
    
    Args:
        K (int): Information block size
        R (float): Code rate
        security_level (int): Kyber security level (512, 768, 1024)
    
    Returns:
        KyMLPLDPCSystem: Initialized system
    """
    print("\n" + "=" * 70)
    print("SYSTEM INITIALIZATION")
    print("=" * 70)
    
    # Initialize Kyber
    print(f"\nInitializing Kyber-{security_level}...")
    kyber = KyberWrapper(security_level)
    kyber.generate_keys()
    print(f"  Public key: {len(kyber.pk)} bytes")
    print(f"  Secret key: {len(kyber.sk)} bytes")
    
    # Initialize preprocessor
    print(f"\nInitializing preprocessor (K={K})...")
    preprocessor = EncryptedPreprocessor(kyber, K=K)
    
    # Initialize LDPC
    print(f"\nInitializing LDPC (K={K}, R={R})...")
    ldpc = initialize_ldpc(K, R)
    
    # Create complete system
    system = KyMLPLDPCSystem(ldpc, kyber, preprocessor)
    
    print("\n" + "=" * 70)
    print("SYSTEM READY")
    print("=" * 70 + "\n")
    
    return system


def run_file_processing():
    """
    Run end-to-end file processing experiment
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: END-TO-END FILE PROCESSING")
    print("=" * 70)
    
    # Get input file
    print("\nSelect input file type:")
    print("1. Image file")
    print("2. Text/signal file")
    choice = input("Enter choice (1/2): ")
    
    if choice == "1":
        data_type = "image"
        default_path = "data/test.png"
    else:
        data_type = "signal"
        default_path = "data/test.txt"
    
    input_path = input(f"Enter input file path (default: {default_path}): ").strip()
    if not input_path:
        input_path = default_path
    
    if not os.path.exists(input_path):
        print(f"Error: File not found: {input_path}")
        return
    
    # Get output path
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    basename = os.path.basename(input_path)
    name, ext = os.path.splitext(basename)
    output_path = os.path.join(output_dir, f"{name}_decoded{ext}")
    
    # Get simulation parameters
    print("\nSimulation parameters:")
    EbNo_dB = float(input("Enter Eb/N0 in dB (default: 3.0): ") or "3.0")
    iteration = int(input("Enter LDPC iterations (default: 1): ") or "1")
    
    # Setup system
    system = setup_system()
    
    # Run processing
    stats = system.process_file_end_to_end(
        input_path, output_path,
        EbNo_dB=EbNo_dB,
        iteration=iteration,
        data_type=data_type
    )
    
    # Display results
    print("\nRESULTS:")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")
    print(f"  BER: {stats['ber']:.10f}")
    print(f"  Time: {stats['processing_time']:.2f} seconds")
    
    # Check if errors occurred
    if stats['ber'] > 0:
        print("\n" + "!" * 70)
        print("WARNING: Bit errors detected!")
        print("!" * 70)
        print("To improve performance, try:")
        print("  1. Increase Eb/N0 (SNR) - Higher SNR = less noise")
        print("  2. Increase LDPC iterations - More iterations = better decoding")
        print("  3. Use lower code rate R (e.g., 1/2 instead of 2/3)")
        print("!" * 70)


def run_ber_simulation():
    """
    Run BER vs SNR simulation experiment
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: BER vs SNR SIMULATION")
    print("=" * 70)
    
    # Get parameters
    print("\nSimulation parameters:")
    K = int(input("Enter K (default: 256): ") or "256")
    R = float(input("Enter code rate R (default: 0.667): ") or "0.667")
    
    num_blocks = int(input("Enter number of test blocks (default: 100): ") or "100")
    
    snr_min = float(input("Enter min Eb/N0 in dB (default: 1.0): ") or "1.0")
    snr_max = float(input("Enter max Eb/N0 in dB (default: 3.0): ") or "3.0")
    snr_points = int(input("Enter number of SNR points (default: 10): ") or "10")
    
    iteration = int(input("Enter LDPC iterations (default: 1): ") or "1")
    
    # Generate random test messages
    print(f"\nGenerating {num_blocks} random {K}-bit messages...")
    msg_list = [np.random.randint(0, 2, K, dtype=np.uint8) for _ in range(num_blocks)]
    
    # Setup system
    system = setup_system(K=K, R=R)
    
    # Run simulation
    EbNo_range = np.linspace(snr_min, snr_max, snr_points)
    snr_list, ber_list = system.run_ber_simulation(
        msg_list, EbNo_range,
        iteration=iteration,
        max_errors=100,
        max_blocks=num_blocks
    )
    
    # Display results
    print("\nRESULTS:")
    print("-" * 70)
    print(f"{'Eb/N0 (dB)':<15}{'BER':<20}")
    print("-" * 70)
    for snr, ber in zip(snr_list, ber_list):
        print(f"{snr:<15.2f}{ber:<20.10f}")
    print("-" * 70)
    
    # Save results
    output_file = f"output/ber_results_K{K}_R{R:.3f}_iter{iteration}.txt"
    with open(output_file, 'w') as f:
        f.write("Eb/N0(dB),BER\n")
        for snr, ber in zip(snr_list, ber_list):
            f.write(f"{snr:.2f},{ber:.15f}\n")
    
    print(f"\nResults saved to: {output_file}")


def run_comparison_test():
    """
    Compare with/without encryption overhead
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: ENCRYPTION OVERHEAD COMPARISON")
    print("=" * 70)
    
    K = 256
    R = 2/3
    num_blocks = 50
    EbNo_dB = 2.5
    
    print(f"\nTest configuration:")
    print(f"  K = {K} bits")
    print(f"  R = {R}")
    print(f"  Blocks = {num_blocks}")
    print(f"  Eb/N0 = {EbNo_dB} dB")
    
    # Generate test messages
    msg_list = [np.random.randint(0, 2, K, dtype=np.uint8) for _ in range(num_blocks)]
    
    # Setup system
    system = setup_system(K=K, R=R)
    
    # Test with encryption (already integrated)
    print("\nTesting with Kyber encryption...")
    import time
    start = time.time()
    
    total_errors_enc = 0
    for msg in msg_list:
        # Encrypt
        ciphertext, encrypted = system.kyber.encrypt_block(msg)
        encrypted_bits = np.unpackbits(np.frombuffer(encrypted, dtype=np.uint8))
        
        # Encode
        encoded = system.ldpc.encode(encrypted_bits)
        
        # Channel
        received = system.transmit_through_channel(encoded, EbNo_dB, R)
        
        # Decode
        decoded = system.ldpc.decode(received, iteration=1)
        
        # Decrypt
        decrypted_bytes = np.packbits(decoded).tobytes()
        decrypted = system.kyber.decrypt_block(ciphertext, decrypted_bytes)
        
        # Count errors
        errors = np.sum(msg != decrypted)
        total_errors_enc += errors
    
    time_enc = time.time() - start
    ber_enc = total_errors_enc / (K * num_blocks)
    
    # Test without encryption
    print("Testing without encryption...")
    start = time.time()
    
    total_errors_plain = 0
    for msg in msg_list:
        # Encode
        encoded = system.ldpc.encode(msg)
        
        # Channel
        received = system.transmit_through_channel(encoded, EbNo_dB, R)
        
        # Decode
        decoded = system.ldpc.decode(received, iteration=1)
        
        # Count errors
        errors = np.sum(msg != decoded)
        total_errors_plain += errors
    
    time_plain = time.time() - start
    ber_plain = total_errors_plain / (K * num_blocks)
    
    # Display comparison
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print(f"{'Metric':<30}{'With Encryption':<20}{'Without Encryption':<20}")
    print("-" * 70)
    print(f"{'Bit errors:':<30}{total_errors_enc:<20}{total_errors_plain:<20}")
    print(f"{'BER:':<30}{ber_enc:<20.10f}{ber_plain:<20.10f}")
    print(f"{'Processing time (s):':<30}{time_enc:<20.3f}{time_plain:<20.3f}")
    print(f"{'Time overhead:':<30}{(time_enc/time_plain - 1)*100:<20.1f}%")
    print("=" * 70)


def main():
    """
    Main entry point
    """
    print("\n" + "=" * 70)
    print("KyMLP-LDPC SYSTEM - MAIN MENU")
    print("=" * 70)
    
    while True:
        print("\nSelect experiment:")
        print("1. End-to-end file processing (image/signal)")
        print("2. BER vs SNR simulation")
        print("3. Encryption overhead comparison")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            run_file_processing()
        elif choice == "2":
            run_ber_simulation()
        elif choice == "3":
            run_comparison_test()
        elif choice == "4":
            print("\nExiting...")
            break
        else:
            print("Invalid choice. Please try again.")
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()