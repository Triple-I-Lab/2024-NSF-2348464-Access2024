"""
Complete KyMLP-LDPC System Integration
Combines Kyber encryption + MLP-LDPC encoding/decoding
Based on the paper's system architecture
"""

import numpy as np
import time
from decimal import Decimal, getcontext


class KyMLPLDPCSystem:
    """
    Complete KyMLP-LDPC system integrating:
    1. Kyber post-quantum encryption
    2. MLP-LDPC encoding/decoding
    3. AWGN channel simulation
    """
    
    def __init__(self, ldpc_encoder, kyber_wrapper, preprocessor):
        """
        Initialize the complete system
        
        Args:
            ldpc_encoder: LDPC encoder/decoder instance
            kyber_wrapper: KyberWrapper instance with keys
            preprocessor: EncryptedPreprocessor instance
        """
        self.ldpc = ldpc_encoder
        self.kyber = kyber_wrapper
        self.preprocessor = preprocessor
        
    def encode_block(self, block):
        """
        Encode a single K-bit block with LDPC
        
        Args:
            block (numpy.ndarray): K-bit block
        
        Returns:
            numpy.ndarray: Encoded bits
        """
        return self.ldpc.encode(block)
    
    def decode_block(self, received_signal, iteration=1):
        """
        Decode a single block with MLP-LDPC
        
        Args:
            received_signal (numpy.ndarray): Received LLR values
            iteration (int): Number of decoding iterations
        
        Returns:
            numpy.ndarray: Decoded K-bit block
        """
        return self.ldpc.decode(received_signal, iteration)
    
    def transmit_through_channel(self, encoded_bits, EbNo_dB, R):
        """
        Simulate BPSK modulation + AWGN channel transmission
        
        Args:
            encoded_bits (numpy.ndarray): Encoded bits
            EbNo_dB (float): Eb/N0 in dB
            R (float): Code rate
        
        Returns:
            numpy.ndarray: Received LLR values
        """
        # BPSK modulation: 0->+1, 1->-1
        symbols = 1 - 2 * encoded_bits
        
        # Calculate noise variance
        EbNo = 10 ** (EbNo_dB / 10)
        sigma_square = 1 / (2 * R * EbNo)
        
        # Add AWGN
        noise = np.random.randn(len(symbols)) * np.sqrt(sigma_square)
        received = symbols + noise
        
        return received
    
    def process_file_end_to_end(self, input_path, output_path, 
                                EbNo_dB=3.0, iteration=1, 
                                data_type='auto'):
        """
        Complete end-to-end processing:
        Input -> Encrypt -> Encode -> Channel -> Decode -> Decrypt -> Output
        
        Args:
            input_path (str): Input file path
            output_path (str): Output file path
            EbNo_dB (float): Channel SNR in dB
            iteration (int): LDPC decoding iterations
            data_type (str): 'auto', 'image', or 'signal'
        
        Returns:
            dict: Processing statistics
        """
        print("\n" + "=" * 70)
        print("KyMLP-LDPC END-TO-END PROCESSING")
        print("=" * 70)
        
        start_time = time.time()
        
        # Step 1: Preprocess with encryption
        print("\n[1/5] Preprocessing and Encryption...")
        ldpc_input_blocks, ciphertexts, encrypted_blocks, metadata = \
            self.preprocessor.process_with_encryption(input_path, data_type)
        print(f"  Total blocks: {len(ldpc_input_blocks)}")
        print(f"  Block size: {len(ldpc_input_blocks[0])} bits")
        
        # Step 2: LDPC Encoding
        print("\n[2/5] LDPC Encoding...")
        encoded_blocks = []
        for i, block in enumerate(ldpc_input_blocks):
            encoded = self.encode_block(block)
            encoded_blocks.append(encoded)
            if (i + 1) % 10 == 0:
                print(f"  Encoded {i + 1}/{len(ldpc_input_blocks)} blocks")
        print(f"  Encoded block size: {len(encoded_blocks[0])} bits")
        
        # Step 3: Channel Transmission
        print("\n[3/5] Channel Transmission (AWGN)...")
        print(f"  Eb/N0: {EbNo_dB} dB")
        received_blocks = []
        for encoded in encoded_blocks:
            received = self.transmit_through_channel(
                encoded, EbNo_dB, self.ldpc.R
            )
            received_blocks.append(received)
        print(f"  Transmitted {len(encoded_blocks)} blocks")
        
        # Step 4: LDPC Decoding
        print("\n[4/5] LDPC Decoding...")
        print(f"  Iterations: {iteration}")
        decoded_blocks = []
        total_errors = 0
        for i, received in enumerate(received_blocks):
            decoded = self.decode_block(received, iteration)
            decoded_blocks.append(decoded)
            
            # Count errors
            errors = np.sum(ldpc_input_blocks[i] != decoded)
            total_errors += errors
            
            if (i + 1) % 10 == 0:
                print(f"  Decoded {i + 1}/{len(received_blocks)} blocks")
        
        print(f"  Total bit errors: {total_errors}")
        
        # Step 5: Decrypt and reconstruct
        print("\n[5/5] Decryption and Reconstruction...")
        decrypted_data = self.preprocessor.reconstruct_with_decryption(
            decoded_blocks, ciphertexts, metadata
        )
        
        # Save output
        self.preprocessor.save_data(decrypted_data, output_path, metadata)
        print(f"  Saved to: {output_path}")
        
        end_time = time.time()
        
        # Calculate statistics
        total_bits = metadata['original_bits']
        ber = total_errors / (len(ldpc_input_blocks) * len(ldpc_input_blocks[0]))
        processing_time = end_time - start_time
        
        stats = {
            'total_blocks': len(ldpc_input_blocks),
            'total_bits': total_bits,
            'bit_errors': total_errors,
            'ber': ber,
            'EbNo_dB': EbNo_dB,
            'iterations': iteration,
            'processing_time': processing_time
        }
        
        print("\n" + "=" * 70)
        print("PROCESSING COMPLETE")
        print("=" * 70)
        print(f"Total blocks: {stats['total_blocks']}")
        print(f"Total bits: {stats['total_bits']}")
        print(f"Bit errors: {stats['bit_errors']}")
        print(f"BER: {stats['ber']:.10f}")
        print(f"Processing time: {stats['processing_time']:.2f} seconds")
        print("=" * 70 + "\n")
        
        return stats
    
    def run_ber_simulation(self, msg_list, EbNo_dB_range, iteration=1, 
                          max_errors=100, max_blocks=1000):
        """
        Run BER simulation over SNR range
        
        Args:
            msg_list (list): List of K-bit message blocks
            EbNo_dB_range (numpy.ndarray): Range of Eb/N0 values in dB
            iteration (int): LDPC decoding iterations
            max_errors (int): Stop when reaching this many errors
            max_blocks (int): Maximum blocks to test per SNR
        
        Returns:
            tuple: (EbNo_dB_list, BER_list)
        """
        print("\n" + "=" * 70)
        print("BER SIMULATION")
        print("=" * 70)
        print(f"SNR range: {EbNo_dB_range[0]:.1f} to {EbNo_dB_range[-1]:.1f} dB")
        print(f"Total blocks: {len(msg_list)}")
        print(f"Iterations: {iteration}")
        print("=" * 70 + "\n")
        
        getcontext().prec = 15
        K = len(msg_list[0])
        
        ber_list = []
        
        for idx, EbNo_dB in enumerate(EbNo_dB_range):
            print(f"[{idx + 1}/{len(EbNo_dB_range)}] Testing Eb/N0 = {EbNo_dB:.2f} dB")
            
            total_errors = 0
            total_bits = 0
            blocks_tested = 0
            
            for msg in msg_list[:max_blocks]:
                # Encode
                encoded = self.encode_block(msg)
                
                # Transmit through channel
                received = self.transmit_through_channel(
                    encoded, EbNo_dB, self.ldpc.R
                )
                
                # Decode
                decoded = self.decode_block(received, iteration)
                
                # Count errors
                errors = np.sum(msg != decoded[:K])
                total_errors += errors
                total_bits += K
                blocks_tested += 1
                
                # Stop early if enough errors collected
                if total_errors >= max_errors:
                    break
            
            # Calculate BER
            if total_bits > 0:
                ber = total_errors / total_bits
            else:
                ber = Decimal(0)
            
            ber_list.append(float(ber))
            
            print(f"  Blocks tested: {blocks_tested}")
            print(f"  Bit errors: {total_errors}")
            print(f"  BER: {ber:.10f}\n")
            
            # Stop if perfect transmission
            if total_errors == 0 and blocks_tested >= 10:
                print("  Perfect transmission achieved, stopping simulation\n")
                break
        
        print("=" * 70)
        print("SIMULATION COMPLETE")
        print("=" * 70)
        
        return list(EbNo_dB_range[:len(ber_list)]), ber_list


class LDPCEncoderDecoder:
    """
    Wrapper for LDPC encoding/decoding functions
    Adapts the LDPC_main.py functions to class interface
    """
    
    def __init__(self, K, R, H, A_sparse, B_sparse, C_sparse, D_sparse,
                 kb, Z_c, mbRM, layer_params_dict):
        """
        Initialize LDPC encoder/decoder with precomputed matrices
        
        Args:
            K (int): Information block size
            R (float): Code rate
            H (numpy.ndarray): Parity check matrix
            A_sparse, B_sparse, C_sparse, D_sparse: Submatrices
            kb (int): Base graph parameter
            Z_c (int): Lifting size
            mbRM (int): Number of rows
            layer_params_dict (dict): Layer processing parameters
        """
        self.K = K
        self.R = R
        self.H = H
        self.A_sparse = A_sparse
        self.B_sparse = B_sparse
        self.C_sparse = C_sparse
        self.D_sparse = D_sparse
        self.kb = kb
        self.Z_c = Z_c
        self.mbRM = mbRM
        self.layer_params_dict = layer_params_dict
    
    def encode(self, msg):
        """
        Encode message with LDPC
        
        Args:
            msg (numpy.ndarray): K-bit message
        
        Returns:
            numpy.ndarray: Encoded bits
        """
        from ldpc import LDPC_encoding
        
        encoded_bits = LDPC_encoding(
            msg, self.kb, self.Z_c,
            self.A_sparse, self.B_sparse,
            self.C_sparse, self.D_sparse
        )
        
        return encoded_bits
    
    def decode(self, received_signal, iteration=1):
        """
        Decode received signal with MLP-LDPC
        
        Args:
            received_signal (numpy.ndarray): Received LLR values
            iteration (int): Number of iterations
        
        Returns:
            numpy.ndarray: Decoded K-bit message
        """
        from ldpc import (process_layer, init_node, calculate_function)
        
        # Apply puncturing and shortening
        LLR_received = received_signal.copy()
        LLR_received[:2 * self.Z_c] = 0
        if self.K < self.kb * self.Z_c:
            LLR_received[self.K:self.kb * self.Z_c] = 10000000
        
        # Initialize LLR dict
        LLR_dict = {index + 1: value for index, value in enumerate(LLR_received)}
        
        # Initialize nodes
        variable_node_updates_json = init_node(self.mbRM)
        check_node_updates_json = init_node(self.mbRM)
        
        # Iterative decoding
        for _ in range(iteration):
            for idx in sorted(self.layer_params_dict.keys()):
                params = self.layer_params_dict[idx]
                results = []
                for param in params:
                    LLR_dict_partial = process_layer(
                        *param, check_node_updates_json,
                        variable_node_updates_json, LLR_dict
                    )
                    results.append(LLR_dict_partial)
                for LLR_dict_partial in results:
                    LLR_dict.update(LLR_dict_partial)
        
        # Hard decision
        received_bits = np.zeros(len(LLR_dict), dtype=int)
        for key, value in LLR_dict.items():
            received_bits[key - 1] = calculate_function(value)
        
        return received_bits[:self.K]


# Test case
if __name__ == "__main__":
    print("=" * 70)
    print("KyMLP-LDPC System Test")
    print("=" * 70)
    
    # This test requires the actual LDPC matrices from ldpc.py
    # For demonstration, we show the structure
    
    print("\nTo run complete system test:")
    print("1. Initialize LDPC matrices from ldpc.py")
    print("2. Create KyberWrapper and generate keys")
    print("3. Create EncryptedPreprocessor")
    print("4. Create LDPCEncoderDecoder wrapper")
    print("5. Create KyMLPLDPCSystem")
    print("6. Run end-to-end processing or BER simulation")
    
    print("\nExample usage:")
    print("-" * 70)
    print("""
from kyber_wrapper import KyberWrapper
from preprocess import EncryptedPreprocessor
from kymlp_ldpc import KyMLPLDPCSystem, LDPCEncoderDecoder

# Setup Kyber
kyber = KyberWrapper(512)
kyber.generate_keys()

# Setup preprocessor
preprocessor = EncryptedPreprocessor(kyber, K=256)

# Setup LDPC (from ldpc.py matrices)
ldpc = LDPCEncoderDecoder(K, R, H, A_sparse, B_sparse, 
                          C_sparse, D_sparse, kb, Z_c, 
                          mbRM, layer_params_dict)

# Create complete system
system = KyMLPLDPCSystem(ldpc, kyber, preprocessor)

# Process a file
stats = system.process_file_end_to_end(
    'data/test.png',
    'output/decoded.png',
    EbNo_dB=3.0,
    iteration=1
)

# Or run BER simulation
msg_list = [np.random.randint(0, 2, K) for _ in range(100)]
snr_range = np.linspace(1.0, 3.0, 10)
snr_list, ber_list = system.run_ber_simulation(
    msg_list, snr_range, iteration=1
)
    """)
    print("-" * 70)
    print("\n" + "=" * 70)