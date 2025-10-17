"""
Preprocessing Module for KyMLP-LDPC System
Handles: Input -> Encryption -> Bit Transformation -> Segmentation
Based on the system diagram in the paper
"""

import numpy as np
import os
from PIL import Image


class Preprocessor:
    """
    Preprocessor for KyMLP-LDPC system
    Converts input data to K-bit blocks ready for LDPC encoding
    """
    
    def __init__(self, K=256):
        """
        Initialize preprocessor
        
        Args:
            K (int): Block size in bits for LDPC encoding (default 256)
        """
        self.K = K
    
    def load_image(self, image_path):
        """
        Load image file and convert to binary data
        
        Args:
            image_path (str): Path to image file
        
        Returns:
            numpy.ndarray: Binary array of image data
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        img = Image.open(image_path)
        img_array = np.array(img)
        
        # Flatten and convert to binary
        flat_array = img_array.flatten()
        binary_data = np.unpackbits(flat_array.astype(np.uint8))
        
        return binary_data, img.size, img.mode
    
    def load_signal(self, signal_path):
        """
        Load signal file (text or binary) and convert to binary data
        
        Args:
            signal_path (str): Path to signal file
        
        Returns:
            numpy.ndarray: Binary array of signal data
        """
        if not os.path.exists(signal_path):
            raise FileNotFoundError(f"Signal file not found: {signal_path}")
        
        # Check file extension
        _, ext = os.path.splitext(signal_path)
        
        if ext in ['.txt', '.csv']:
            # Text file - read as string and convert to binary
            with open(signal_path, 'r') as f:
                content = f.read()
            byte_data = content.encode('utf-8')
            binary_data = np.unpackbits(np.frombuffer(byte_data, dtype=np.uint8))
        else:
            # Binary file - read directly
            with open(signal_path, 'rb') as f:
                byte_data = f.read()
            binary_data = np.unpackbits(np.frombuffer(byte_data, dtype=np.uint8))
        
        return binary_data
    
    def load_data(self, file_path, data_type='auto'):
        """
        Load data from file (auto-detect or specify type)
        
        Args:
            file_path (str): Path to data file
            data_type (str): 'auto', 'image', or 'signal'
        
        Returns:
            tuple: (binary_data, metadata)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        _, ext = os.path.splitext(file_path)
        
        # Auto-detect type
        if data_type == 'auto':
            image_exts = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
            if ext.lower() in image_exts:
                data_type = 'image'
            else:
                data_type = 'signal'
        
        # Load based on type
        metadata = {'file_path': file_path, 'type': data_type}
        
        if data_type == 'image':
            binary_data, img_size, img_mode = self.load_image(file_path)
            metadata['image_size'] = img_size
            metadata['image_mode'] = img_mode
        else:
            binary_data = self.load_signal(file_path)
        
        metadata['original_bits'] = len(binary_data)
        
        return binary_data, metadata
    
    def segment_to_blocks(self, binary_data, K=None):
        """
        Segment binary data into K-bit blocks
        Pad last block if necessary
        
        Args:
            binary_data (numpy.ndarray): Binary data array
            K (int): Block size (uses self.K if None)
        
        Returns:
            list: List of K-bit blocks (numpy arrays)
        """
        if K is None:
            K = self.K
        
        n_bits = len(binary_data)
        n_blocks = (n_bits + K - 1) // K
        padded_size = n_blocks * K
        
        # Pad if necessary
        if n_bits < padded_size:
            binary_data = np.pad(binary_data, (0, padded_size - n_bits), 
                                mode='constant', constant_values=0)
        
        # Split into blocks
        blocks = []
        for i in range(n_blocks):
            block = binary_data[i * K:(i + 1) * K]
            blocks.append(block)
        
        return blocks
    
    def blocks_to_data(self, blocks, original_bits):
        """
        Reconstruct binary data from K-bit blocks
        Remove padding based on original size
        
        Args:
            blocks (list): List of K-bit blocks
            original_bits (int): Original data size before padding
        
        Returns:
            numpy.ndarray: Reconstructed binary data
        """
        # Concatenate all blocks
        binary_data = np.concatenate(blocks)
        
        # Remove padding
        binary_data = binary_data[:original_bits]
        
        return binary_data
    
    def save_image(self, binary_data, output_path, image_size, image_mode):
        """
        Save binary data as image
        
        Args:
            binary_data (numpy.ndarray): Binary data
            output_path (str): Output file path
            image_size (tuple): (width, height)
            image_mode (str): Image mode (e.g., 'RGB', 'L')
        """
        # Convert binary to byte array
        byte_data = np.packbits(binary_data)
        
        # Reshape to image dimensions
        if image_mode == 'RGB':
            channels = 3
        elif image_mode == 'RGBA':
            channels = 4
        else:
            channels = 1
        
        width, height = image_size
        expected_size = width * height * channels
        
        # Ensure correct size
        if len(byte_data) > expected_size:
            byte_data = byte_data[:expected_size]
        elif len(byte_data) < expected_size:
            byte_data = np.pad(byte_data, (0, expected_size - len(byte_data)), 
                              mode='constant')
        
        # Reshape and create image
        if channels == 1:
            img_array = byte_data.reshape((height, width))
        else:
            img_array = byte_data.reshape((height, width, channels))
        
        img = Image.fromarray(img_array.astype(np.uint8), mode=image_mode)
        img.save(output_path)
    
    def save_signal(self, binary_data, output_path, encoding='utf-8'):
        """
        Save binary data as signal file
        
        Args:
            binary_data (numpy.ndarray): Binary data
            output_path (str): Output file path
            encoding (str): Text encoding if saving as text
        """
        # Convert binary to bytes
        byte_data = np.packbits(binary_data).tobytes()
        
        _, ext = os.path.splitext(output_path)
        
        if ext in ['.txt', '.csv']:
            # Save as text
            try:
                text = byte_data.decode(encoding)
                with open(output_path, 'w', encoding=encoding) as f:
                    f.write(text)
            except:
                # If decode fails, save as binary
                with open(output_path, 'wb') as f:
                    f.write(byte_data)
        else:
            # Save as binary
            with open(output_path, 'wb') as f:
                f.write(byte_data)
    
    def save_data(self, binary_data, output_path, metadata):
        """
        Save binary data back to file based on metadata
        
        Args:
            binary_data (numpy.ndarray): Binary data
            output_path (str): Output file path
            metadata (dict): Metadata from load_data
        """
        if metadata['type'] == 'image':
            self.save_image(binary_data, output_path, 
                          metadata['image_size'], 
                          metadata['image_mode'])
        else:
            self.save_signal(binary_data, output_path)


class EncryptedPreprocessor(Preprocessor):
    """
    Preprocessor with encryption capability
    Adds encryption step: Input -> Encryption -> Bit Transformation -> Segmentation
    """
    
    def __init__(self, kyber_wrapper, K=256):
        """
        Initialize encrypted preprocessor
        
        Args:
            kyber_wrapper: KyberWrapper instance with generated keys
            K (int): Block size in bits
        """
        super().__init__(K)
        self.kyber = kyber_wrapper
    
    def process_with_encryption(self, file_path, data_type='auto'):
        """
        Complete preprocessing with encryption
        Flow: Input -> Encryption -> Bit Transformation -> Segmentation
        
        Args:
            file_path (str): Path to input file
            data_type (str): 'auto', 'image', or 'signal'
        
        Returns:
            tuple: (blocks, ciphertexts, encrypted_blocks, metadata)
        """
        # Step 1: Load input data
        binary_data, metadata = self.load_data(file_path, data_type)
        print(f"Loaded {len(binary_data)} bits from {file_path}")
        
        # Step 2: Segment into K-bit blocks
        input_blocks = self.segment_to_blocks(binary_data)
        print(f"Segmented into {len(input_blocks)} blocks of {self.K} bits")
        
        # Step 3: Encrypt each block using Kyber
        ciphertexts = []
        encrypted_blocks = []
        
        for i, block in enumerate(input_blocks):
            ciphertext, encrypted_data = self.kyber.encrypt_block(block)
            ciphertexts.append(ciphertext)
            encrypted_blocks.append(encrypted_data)
        
        print(f"Encrypted {len(input_blocks)} blocks using Kyber")
        
        # Step 4: Convert encrypted data back to bit blocks for LDPC
        ldpc_input_blocks = []
        for encrypted_data in encrypted_blocks:
            # Convert encrypted bytes to bits
            encrypted_bits = np.unpackbits(np.frombuffer(encrypted_data, dtype=np.uint8))
            ldpc_input_blocks.append(encrypted_bits)
        
        # Store encryption info in metadata
        metadata['encrypted'] = True
        metadata['n_blocks'] = len(input_blocks)
        
        return ldpc_input_blocks, ciphertexts, encrypted_blocks, metadata
    
    def reconstruct_with_decryption(self, ldpc_output_blocks, ciphertexts, metadata):
        """
        Reconstruct data with decryption
        Flow: De-bit Transformation -> Decryption -> Output
        
        Args:
            ldpc_output_blocks (list): Decoded K-bit blocks from LDPC
            ciphertexts (list): Kyber ciphertexts for decryption
            metadata (dict): Metadata from process_with_encryption
        
        Returns:
            numpy.ndarray: Decrypted and reconstructed binary data
        """
        # Step 1: Convert bit blocks back to encrypted byte blocks
        encrypted_blocks = []
        for bit_block in ldpc_output_blocks:
            encrypted_bytes = np.packbits(bit_block).tobytes()
            encrypted_blocks.append(encrypted_bytes)
        
        print(f"Converting {len(ldpc_output_blocks)} blocks from LDPC output")
        
        # Step 2: Decrypt each block using Kyber
        decrypted_blocks = []
        for ciphertext, encrypted_data in zip(ciphertexts, encrypted_blocks):
            decrypted_bits = self.kyber.decrypt_block(ciphertext, encrypted_data)
            decrypted_blocks.append(decrypted_bits)
        
        print(f"Decrypted {len(encrypted_blocks)} blocks using Kyber")
        
        # Step 3: Reconstruct original data
        binary_data = self.blocks_to_data(decrypted_blocks, metadata['original_bits'])
        print(f"Reconstructed {len(binary_data)} bits of original data")
        
        return binary_data


# Test cases
if __name__ == "__main__":
    print("=" * 60)
    print("Preprocessor Test Cases")
    print("=" * 60)
    
    # Test Case 1: Basic preprocessing without encryption
    print("\nTest Case 1: Basic Preprocessing (No Encryption)")
    print("-" * 60)
    
    # Create test binary data
    test_data = np.random.randint(0, 2, size=1000, dtype=np.uint8)
    print(f"Generated {len(test_data)} bits of test data")
    
    preprocessor = Preprocessor(K=256)
    
    # Segment
    blocks = preprocessor.segment_to_blocks(test_data)
    print(f"Segmented into {len(blocks)} blocks")
    print(f"Each block size: {len(blocks[0])} bits")
    
    # Reconstruct
    reconstructed = preprocessor.blocks_to_data(blocks, len(test_data))
    print(f"Reconstructed {len(reconstructed)} bits")
    
    # Verify
    assert np.array_equal(test_data, reconstructed)
    print("Test Case 1 PASSED: Data correctly segmented and reconstructed")
    
    # Test Case 2: Preprocessing with encryption
    print("\n" + "=" * 60)
    print("Test Case 2: Preprocessing with Encryption")
    print("-" * 60)
    
    from kyber_wrapper import KyberWrapper
    
    # Initialize Kyber
    kyber = KyberWrapper(security_level=512)
    kyber.generate_keys()
    print("Kyber keys generated")
    
    # Create encrypted preprocessor
    enc_preprocessor = EncryptedPreprocessor(kyber, K=256)
    
    # Create test file
    test_file = "test_input.txt"
    with open(test_file, 'w') as f:
        f.write("This is a test message for preprocessing!")
    
    # Process with encryption
    ldpc_blocks, ciphertexts, encrypted_blocks, metadata = \
        enc_preprocessor.process_with_encryption(test_file)
    
    print(f"\nProcessing complete:")
    print(f"  LDPC input blocks: {len(ldpc_blocks)}")
    print(f"  Each block size: {len(ldpc_blocks[0])} bits")
    print(f"  Ciphertexts: {len(ciphertexts)}")
    print(f"  Original bits: {metadata['original_bits']}")
    
    # Simulate LDPC encoding/decoding (identity for test)
    ldpc_output_blocks = ldpc_blocks.copy()
    
    # Reconstruct with decryption
    decrypted_data = enc_preprocessor.reconstruct_with_decryption(
        ldpc_output_blocks, ciphertexts, metadata
    )
    
    # Save and verify
    output_file = "test_output.txt"
    enc_preprocessor.save_data(decrypted_data, output_file, metadata)
    print(f"\nSaved to {output_file}")
    
    # Read back and compare
    with open(test_file, 'r') as f:
        original = f.read()
    with open(output_file, 'r') as f:
        decrypted = f.read()
    
    assert original == decrypted
    print("Test Case 2 PASSED: End-to-end encryption/decryption successful")
    
    # Cleanup
    os.remove(test_file)
    os.remove(output_file)
    
    print("\n" + "=" * 60)
    print("All Test Cases PASSED")
    print("=" * 60)