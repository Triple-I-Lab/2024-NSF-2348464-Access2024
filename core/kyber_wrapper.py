"""
Kyber Post-Quantum Cryptography Wrapper
Implements encryption/decryption for KyMLP-LDPC system using Kyber KEM
Based on kyber-py library
"""

import numpy as np
from kyber_py.kyber import Kyber512, Kyber768, Kyber1024


class KyberWrapper:
    """
    Wrapper for Kyber post-quantum cryptography (KEM)
    Supports Kyber-512, Kyber-768, and Kyber-1024
    
    Note: Kyber is a KEM (Key Encapsulation Mechanism), not direct encryption.
    We use the shared secret as a deterministic key for XOR encryption.
    """
    
    SECURITY_LEVELS = {
        512: Kyber512,
        768: Kyber768,
        1024: Kyber1024
    }
    
    def __init__(self, security_level=512):
        """
        Initialize Kyber with specified security level
        
        Args:
            security_level (int): 512, 768, or 1024 bits
        """
        if security_level not in self.SECURITY_LEVELS:
            raise ValueError(f"Security level must be one of {list(self.SECURITY_LEVELS.keys())}")
        
        self.security_level = security_level
        self.kyber = self.SECURITY_LEVELS[security_level]
        self.pk = None
        self.sk = None
        
    def generate_keys(self):
        """
        Generate public and private key pair
        
        Returns:
            tuple: (public_key, secret_key)
        """
        self.pk, self.sk = self.kyber.keygen()
        return self.pk, self.sk
    
    def encrypt(self, data):
        """
        Encrypt data using Kyber KEM
        
        Process:
        1. Generate shared secret (key) and ciphertext (c) using pk
        2. XOR data with shared secret for encryption
        3. Return (ciphertext, encrypted_data)
        
        Args:
            data (bytes): Data to encrypt (32 bytes = 256 bits)
        
        Returns:
            tuple: (ciphertext, encrypted_data)
                - ciphertext: Kyber ciphertext to reconstruct shared secret
                - encrypted_data: XOR(data, shared_secret)
        """
        if self.pk is None:
            raise RuntimeError("Keys not generated. Call generate_keys() first.")
        
        # Convert numpy array to bytes if needed
        if isinstance(data, np.ndarray):
            data = data.tobytes()
        
        # Kyber generates 32 bytes (256 bits) shared secret
        if len(data) != 32:
            raise ValueError(f"Data must be 32 bytes (256 bits), got {len(data)} bytes")
        
        # Encapsulate: generate shared secret and ciphertext
        shared_secret, ciphertext = self.kyber.encaps(self.pk)
        
        # XOR data with shared secret for encryption
        data_array = np.frombuffer(data, dtype=np.uint8)
        key_array = np.frombuffer(shared_secret, dtype=np.uint8)
        encrypted_data = np.bitwise_xor(data_array, key_array).tobytes()
        
        return ciphertext, encrypted_data
    
    def decrypt(self, ciphertext, encrypted_data):
        """
        Decrypt data using Kyber KEM
        
        Process:
        1. Decapsulate ciphertext to recover shared secret using sk
        2. XOR encrypted_data with shared secret to recover original data
        
        Args:
            ciphertext (bytes): Kyber ciphertext
            encrypted_data (bytes): XOR-encrypted data
        
        Returns:
            bytes: Decrypted original data (32 bytes)
        """
        if self.sk is None:
            raise RuntimeError("Keys not generated. Call generate_keys() first.")
        
        # Decapsulate: recover shared secret from ciphertext
        shared_secret = self.kyber.decaps(self.sk, ciphertext)
        
        # XOR encrypted data with shared secret to recover original
        encrypted_array = np.frombuffer(encrypted_data, dtype=np.uint8)
        key_array = np.frombuffer(shared_secret, dtype=np.uint8)
        decrypted_data = np.bitwise_xor(encrypted_array, key_array).tobytes()
        
        return decrypted_data
    
    def encrypt_block(self, block):
        """
        Encrypt a single block of K=256 bits
        
        Args:
            block (numpy.ndarray): Binary array of length 256
        
        Returns:
            tuple: (ciphertext, encrypted_data)
        """
        if len(block) != 256:
            raise ValueError(f"Block must be 256 bits, got {len(block)} bits")
        
        # Convert bit array to bytes (256 bits = 32 bytes)
        block_bytes = np.packbits(block).tobytes()
        
        return self.encrypt(block_bytes)
    
    def decrypt_block(self, ciphertext, encrypted_data):
        """
        Decrypt ciphertext back to 256-bit block
        
        Args:
            ciphertext (bytes): Kyber ciphertext
            encrypted_data (bytes): XOR-encrypted data
        
        Returns:
            numpy.ndarray: Binary array of length 256
        """
        plaintext_bytes = self.decrypt(ciphertext, encrypted_data)
        
        # Convert bytes back to bit array
        plaintext_bits = np.unpackbits(np.frombuffer(plaintext_bytes, dtype=np.uint8))
        
        return plaintext_bits
    
    def get_ciphertext_size(self):
        """
        Get the size of Kyber ciphertext for this security level
        
        Returns:
            int: Ciphertext size in bytes
        """
        # Kyber ciphertext sizes (from kyber-py documentation)
        sizes = {
            512: 768,   # Kyber-512: 768 bytes ciphertext
            768: 1088,  # Kyber-768: 1088 bytes ciphertext
            1024: 1568  # Kyber-1024: 1568 bytes ciphertext
        }
        return sizes[self.security_level]


def encrypt_data_blocks(data_bits, kyber_wrapper, block_size=256):
    """
    Utility function to encrypt data split into K=256 bit blocks
    As described in the paper (Section III.B, Algorithm 3)
    
    Args:
        data_bits (numpy.ndarray): Binary data to encrypt
        kyber_wrapper (KyberWrapper): Initialized Kyber wrapper with keys
        block_size (int): Block size in bits (default 256 for paper)
    
    Returns:
        tuple: (ciphertexts, encrypted_blocks, original_size)
            - ciphertexts: List of Kyber ciphertexts
            - encrypted_blocks: List of XOR-encrypted data blocks
            - original_size: Original data size before padding
    """
    # Pad data to multiple of block_size
    n_bits = len(data_bits)
    n_blocks = (n_bits + block_size - 1) // block_size
    padded_size = n_blocks * block_size
    
    if n_bits < padded_size:
        data_bits = np.pad(data_bits, (0, padded_size - n_bits), mode='constant')
    
    # Split into blocks and encrypt each
    ciphertexts = []
    encrypted_blocks = []
    
    for i in range(n_blocks):
        block = data_bits[i * block_size:(i + 1) * block_size]
        ciphertext, encrypted_data = kyber_wrapper.encrypt_block(block)
        ciphertexts.append(ciphertext)
        encrypted_blocks.append(encrypted_data)
    
    return ciphertexts, encrypted_blocks, n_bits


def decrypt_data_blocks(ciphertexts, encrypted_blocks, kyber_wrapper, original_size):
    """
    Utility function to decrypt blocks back to original data
    
    Args:
        ciphertexts (list): List of Kyber ciphertexts
        encrypted_blocks (list): List of XOR-encrypted data blocks
        kyber_wrapper (KyberWrapper): Initialized Kyber wrapper with keys
        original_size (int): Original data size before padding
    
    Returns:
        numpy.ndarray: Decrypted binary data
    """
    decrypted_blocks = []
    
    for ciphertext, encrypted_data in zip(ciphertexts, encrypted_blocks):
        block = kyber_wrapper.decrypt_block(ciphertext, encrypted_data)
        decrypted_blocks.append(block)
    
    # Concatenate all blocks and remove padding
    decrypted_data = np.concatenate(decrypted_blocks)
    decrypted_data = decrypted_data[:original_size]
    
    return decrypted_data


# Test cases
if __name__ == "__main__":
    print("=" * 60)
    print("Kyber Wrapper Test Cases")
    print("=" * 60)
    
    # Test Case 1: Basic encryption/decryption with Kyber-512
    print("\nTest Case 1: Basic Encryption/Decryption (Kyber-512 KEM)")
    print("-" * 60)
    
    # Initialize Kyber-512
    kyber = KyberWrapper(security_level=512)
    pk, sk = kyber.generate_keys()
    print(f" Keys generated")
    print(f"  Public key size: {len(pk)} bytes")
    print(f"  Secret key size: {len(sk)} bytes")
    
    # Create test data (256 bits = 32 bytes)
    test_message = b"This is a test message 32b!!!!!!"  # Exactly 32 bytes
    assert len(test_message) == 32, f"Expected 32 bytes, got {len(test_message)}"
    print(f"\n Original message: {test_message}")
    
    # Encrypt
    ciphertext, encrypted_data = kyber.encrypt(test_message)
    print(f" Encrypted successfully")
    print(f"  Kyber ciphertext size: {len(ciphertext)} bytes")
    print(f"  Encrypted data size: {len(encrypted_data)} bytes")
    
    # Decrypt
    decrypted = kyber.decrypt(ciphertext, encrypted_data)
    print(f" Decrypted successfully")
    print(f"  Decrypted message: {decrypted}")
    
    # Verify
    assert test_message == decrypted, "Decryption failed!"
    print(f"\n Test Case 1 PASSED: Message successfully encrypted and decrypted")
    
    
    # Test Case 2: Block-based encryption for 256-bit blocks (as in paper)
    print("\n" + "=" * 60)
    print("Test Case 2: Block-based Encryption (Paper's K=256 approach)")
    print("-" * 60)
    
    # Create random binary data (3 blocks of 256 bits each = 768 bits total)
    np.random.seed(42)
    original_data = np.random.randint(0, 2, size=768, dtype=np.uint8)
    print(f" Generated random data: {len(original_data)} bits")
    print(f"  First 32 bits: {original_data[:32]}")
    
    # Encrypt blocks
    kyber2 = KyberWrapper(security_level=512)
    kyber2.generate_keys()
    
    ciphertexts, encrypted_blocks, original_size = encrypt_data_blocks(original_data, kyber2)
    print(f"\n Encrypted {len(ciphertexts)} blocks")
    print(f"  Each Kyber ciphertext: {len(ciphertexts[0])} bytes")
    print(f"  Each encrypted block: {len(encrypted_blocks[0])} bytes")
    print(f"  Total ciphertext size: {sum(len(c) for c in ciphertexts)} bytes")
    print(f"  Total encrypted data: {sum(len(e) for e in encrypted_blocks)} bytes")
    
    # Decrypt blocks
    decrypted_data = decrypt_data_blocks(ciphertexts, encrypted_blocks, kyber2, original_size)
    print(f"\n Decrypted {len(ciphertexts)} blocks")
    print(f"  Decrypted data size: {len(decrypted_data)} bits")
    print(f"  First 32 bits: {decrypted_data[:32]}")
    
    # Verify
    assert len(original_data) == len(decrypted_data), "Size mismatch!"
    assert np.array_equal(original_data, decrypted_data), "Data mismatch!"
    print(f"\n Test Case 2 PASSED: All {len(ciphertexts)} blocks correctly encrypted/decrypted")
    
    
    # Summary
    print("\n" + "=" * 60)
    print("All Test Cases PASSED ")
    print("=" * 60)
    print(f"Kyber-512 overhead per 32-byte message:")
    print(f"  Ciphertext: {kyber.get_ciphertext_size()} bytes")
    print(f"  Encrypted data: 32 bytes")
    print(f"  Total: {kyber.get_ciphertext_size() + 32} bytes")
    print(f"  Expansion ratio: {(kyber.get_ciphertext_size() + 32) / 32:.1f}x")