# -*- coding: utf-8 -*-
"""
Homomorphic Encryption (HE) Breast Cancer Inference
====================================================
Architecture:
  - Hospital (client): holds patient data + secret key
  - Model server     : holds trained weights, NEVER sees plaintext
  - Inference        : runs entirely on encrypted ciphertext

Two modes controlled by USE_TENSEAL flag:
  False -> pure-Python simulation (same API, no install needed)
  True  -> real CKKS homomorphic encryption via TenSEAL

Install for real HE:
    pip install tenseal torch scikit-learn pandas numpy
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings("ignore")

USE_TENSEAL = False   # ← flip to True after: pip install tenseal

# ──────────────────────────────────────────────────────────────
# 1.  DATA LOADING  (same as your original notebook)
# ──────────────────────────────────────────────────────────────

def load_data(csv_path: str = "breastcancer.csv"):
    df = pd.read_csv(csv_path)
    df.drop(columns=["id"], inplace=True, errors="ignore")

    # target in column 0, features in columns 1+
    X = df.iloc[:, 1:].values.astype(np.float64)
    y = df.iloc[:, 0].values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y).astype(np.float64)   # M=1, B=0 (or vice-versa)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler, encoder


# ──────────────────────────────────────────────────────────────
# 2.  PLAINTEXT MODEL WITH POLYNOMIAL ACTIVATION
#     Replaces sigmoid with a degree-3 polynomial that can be
#     evaluated homomorphically:
#       poly_sigmoid(x) ≈ 0.5 + 0.197x - 0.004x³
#     This approximation is accurate for x ∈ [-5, 5].
# ──────────────────────────────────────────────────────────────

def poly_sigmoid(z: torch.Tensor) -> torch.Tensor:
    """Degree-3 polynomial approximation of sigmoid, HE-compatible."""
    return 0.5 + 0.197 * z - 0.004 * (z ** 3)


class PlaintextHECompatibleModel(nn.Module):
    """
    Single linear layer + polynomial activation.
    Identical structure to your original model but the activation
    is polynomial so weights transfer directly to HE inference.
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.linear = nn.Linear(num_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z    = self.linear(x)
        ypred = poly_sigmoid(z)
        return ypred


def train_plaintext_model(X_train, y_train, epochs=50, lr=0.05):
    num_features = X_train.shape[1]
    model = PlaintextHECompatibleModel(num_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.BCELoss()

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    print("Training plaintext model (polynomial activation)...")
    for epoch in range(epochs):
        model.train()
        pred = model(X_t)
        # clamp to valid BCE range due to polynomial approximation
        pred = pred.clamp(1e-7, 1 - 1e-7)
        loss = loss_fn(pred, y_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}  Loss: {loss.item():.4f}")

    # extract weights as numpy for the HE server
    W = model.linear.weight.detach().numpy().flatten()   # shape (num_features,)
    b = model.linear.bias.detach().numpy().item()        # scalar
    print(f"\nModel trained. Weights shape: {W.shape}, bias: {b:.4f}")
    return model, W, b


def evaluate_plaintext(model, X_test, y_test, threshold=0.5):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32)
        pred = model(X_t).numpy().flatten()
    y_pred = (pred > threshold).astype(int)
    acc = (y_pred == y_test.astype(int)).mean()
    print(f"Plaintext accuracy: {acc * 100:.2f}%")
    return acc


# ──────────────────────────────────────────────────────────────
# 3A.  SIMULATED HE  (no external library required)
#      Wraps numpy arrays in an object that mimics TenSEAL's API
#      so you can develop and test without installing TenSEAL.
# ──────────────────────────────────────────────────────────────

class SimulatedCiphertext:
    """
    Mimics tenseal.CKKSVector.
    In simulation, the 'ciphertext' is just the plaintext vector
    stored internally — the model server never calls .plaintext().
    """
    def __init__(self, plaintext_vector: np.ndarray):
        self._data = plaintext_vector.copy()   # secret internal storage

    # ---- operations the server is allowed to call ----
    def add(self, other):
        if isinstance(other, SimulatedCiphertext):
            return SimulatedCiphertext(self._data + other._data)
        return SimulatedCiphertext(self._data + other)

    def mul_plain(self, scalar_or_vector):
        return SimulatedCiphertext(self._data * scalar_or_vector)

    def dot(self, weight_vector: np.ndarray) -> "SimulatedCiphertext":
        """Encrypted dot product with plaintext weights."""
        result = np.dot(self._data, weight_vector)
        return SimulatedCiphertext(np.array([result]))

    def square(self) -> "SimulatedCiphertext":
        return SimulatedCiphertext(self._data ** 2)

    def __add__(self, other):
        return self.add(other)

    def __mul__(self, other):
        return self.mul_plain(other)

    def __rmul__(self, other):
        return self.mul_plain(other)

    # ---- only the CLIENT (key holder) may call this ----
    def decrypt(self) -> np.ndarray:
        return self._data.copy()


class SimulatedHEContext:
    """Mimics tenseal.context() — holds keys, created by client."""
    def __init__(self):
        print("  [SIM] Created simulated HE context (CKKS-like).")

    def encrypt(self, vector: np.ndarray) -> SimulatedCiphertext:
        return SimulatedCiphertext(vector)


# ──────────────────────────────────────────────────────────────
# 3B.  REAL TENSEAL HE
#      Uncomment and use when tenseal is installed.
#      The server-side functions are identical — only the
#      context creation and encrypt/decrypt differ.
# ──────────────────────────────────────────────────────────────

def create_real_context():
    """
    Create a real CKKS context with TenSEAL.
    Call this ONLY on the client (hospital) side.
    """
    try:
        import tenseal as ts
        ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        ctx.generate_galois_keys()
        ctx.global_scale = 2 ** 40
        print("  [REAL] TenSEAL CKKS context created.")
        return ctx
    except ImportError:
        raise ImportError("Install TenSEAL: pip install tenseal")


def real_encrypt(context, vector: np.ndarray):
    """Encrypt a numpy vector using TenSEAL CKKS."""
    import tenseal as ts
    return ts.ckks_vector(context, vector.tolist())


def real_decrypt(encrypted_result) -> np.ndarray:
    """Decrypt the result (client-side only)."""
    return np.array(encrypted_result.decrypt())


# ──────────────────────────────────────────────────────────────
# 4.  HE SERVER  —  all operations run on encrypted data
#     The server receives: encrypted feature vector + plaintext W, b
#     The server NEVER calls .decrypt()
# ──────────────────────────────────────────────────────────────

def he_linear_layer(enc_x, W: np.ndarray, b: float):
    """
    Compute z = W·x + b entirely on ciphertext.
    Works identically for SimulatedCiphertext and ts.CKKSVector.

    Parameters
    ----------
    enc_x : encrypted feature vector (SimulatedCiphertext or CKKSVector)
    W     : plaintext weight vector, shape (num_features,)
    b     : plaintext bias scalar
    """
    # dot product: Σ wᵢ·xᵢ  (encrypted)
    enc_z = enc_x.dot(W)
    # add bias (plaintext scalar)
    enc_z = enc_z + b
    return enc_z


def he_poly_sigmoid(enc_z):
    """
    Polynomial approximation of sigmoid, evaluated on ciphertext:
        σ(z) ≈ 0.5 + 0.197z - 0.004z³
    Only uses + and ×, which are natively supported by HE.
    """
    # term1 = 0.5  (constant)
    # term2 = 0.197 * z
    # term3 = -0.004 * z³  =  -0.004 * z * z * z  (3 multiplications)
    enc_z2   = enc_z.square()                    # z²
    enc_z3   = enc_z2 * enc_z if hasattr(enc_z2, '__mul__') else enc_z2.mul_plain(enc_z._data)
    # For TenSEAL: enc_z3 = enc_z2 * enc_z  works directly
    # For our simulator we need a small workaround:
    enc_term2 = enc_z * 0.197
    enc_term3 = enc_z3 * (-0.004)
    enc_result = enc_term2 + enc_term3 + 0.5
    return enc_result


def he_inference(enc_x, W: np.ndarray, b: float):
    """
    Full encrypted inference pipeline (server side).
    Returns an encrypted prediction that only the client can decrypt.
    """
    enc_z    = he_linear_layer(enc_x, W, b)
    enc_pred = he_poly_sigmoid(enc_z)
    return enc_pred


# ──────────────────────────────────────────────────────────────
# 5.  CLIENT  —  encrypt, send, receive, decrypt
# ──────────────────────────────────────────────────────────────

class HospitalClient:
    """
    Represents the hospital.
    Holds the secret key and patient data.
    Only component that may call .decrypt().
    """
    def __init__(self, use_real_he: bool = False):
        self.use_real = use_real_he
        if use_real_he:
            self.context = create_real_context()
        else:
            self.context = SimulatedHEContext()

    def encrypt_patient(self, feature_vector: np.ndarray):
        """Encrypt a single patient's feature vector."""
        if self.use_real:
            return real_encrypt(self.context, feature_vector)
        return self.context.encrypt(feature_vector)

    def decrypt_prediction(self, encrypted_result) -> float:
        """Decrypt the inference result."""
        if self.use_real:
            val = real_decrypt(encrypted_result)
        else:
            val = encrypted_result.decrypt()
        return float(val[0])

    def diagnose(self, raw_prediction: float, threshold: float = 0.5,
                 encoder: LabelEncoder = None) -> str:
        label_idx = int(raw_prediction > threshold)
        if encoder is not None:
            label = encoder.inverse_transform([label_idx])[0]
            return f"{'MALIGNANT' if label == 'M' else 'BENIGN'} (confidence: {raw_prediction:.4f})"
        return f"{'Cancer likely' if label_idx == 1 else 'Likely benign'} ({raw_prediction:.4f})"


# ──────────────────────────────────────────────────────────────
# 6.  END-TO-END DEMO
# ──────────────────────────────────────────────────────────────

def run_demo(csv_path: str = "breastcancer.csv"):
    print("=" * 60)
    print("  HE Breast Cancer Inference Demo")
    print(f"  Mode: {'Real TenSEAL CKKS' if USE_TENSEAL else 'Simulated HE (dev mode)'}")
    print("=" * 60)

    # ── Step 1: Load data ──
    print("\n[1] Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, scaler, encoder = load_data(csv_path)
    print(f"    Train: {X_train.shape}, Test: {X_test.shape}")

    # ── Step 2: Train plaintext model ──
    print("\n[2] Training plaintext model (hospital trains, then hands weights to server)...")
    model, W, b = train_plaintext_model(X_train, y_train, epochs=60)
    evaluate_plaintext(model, X_test, y_test)

    # ── Step 3: Hospital creates HE context ──
    print("\n[3] Hospital initialising HE context and key pair...")
    client = HospitalClient(use_real_he=USE_TENSEAL)

    # ── Step 4: Encrypted batch inference ──
    print("\n[4] Running encrypted inference on test set...")
    correct = 0
    n = len(X_test)

    for i in range(n):
        # CLIENT: encrypt this patient's data
        enc_features = client.encrypt_patient(X_test[i])

        # SERVER: run inference on ciphertext (never sees plaintext)
        enc_prediction = he_inference(enc_features, W, b)

        # CLIENT: decrypt the result
        raw_pred = client.decrypt_prediction(enc_prediction)
        predicted_label = int(raw_pred > 0.5)
        correct += int(predicted_label == int(y_test[i]))

        if i < 5:
            diagnosis = client.diagnose(raw_pred, encoder=encoder)
            true_label = encoder.inverse_transform([int(y_test[i])])[0]
            print(f"    Patient {i+1:3d}: {diagnosis} | True: {true_label}")

    he_accuracy = correct / n
    print(f"\n[5] HE inference accuracy: {he_accuracy * 100:.2f}%  ({correct}/{n} correct)")
    print("\nNote: small accuracy gap vs plaintext is due to poly sigmoid approximation.")
    print("      For higher accuracy, use a degree-5 polynomial approximation.")

    return he_accuracy


# ──────────────────────────────────────────────────────────────
# 7.  UPGRADE PATH TO REAL TENSEAL
# ──────────────────────────────────────────────────────────────

TENSEAL_UPGRADE_GUIDE = """
TO USE REAL HOMOMORPHIC ENCRYPTION:
────────────────────────────────────
1.  pip install tenseal

2.  Set USE_TENSEAL = True at the top of this file.

3.  The only changes to the pipeline are:
    - context creation   →  create_real_context()
    - encrypt()          →  ts.ckks_vector(context, vector)
    - decrypt()          →  encrypted.decrypt()
    - he_inference()     →  unchanged (CKKS vectors support + and *)

4.  For production, the server receives ONLY:
    - The serialised public evaluation context (no secret key)
    - The encrypted feature vectors

    context.make_context_public()        # strip secret key before sending
    enc_x_bytes = enc_x.serialize()      # serialise for network transfer
    enc_x = ts.ckks_vector_from(ctx, enc_x_bytes)  # deserialise on server

5.  CKKS parameter guidance:
    poly_modulus_degree = 8192  for 1-layer models  (fast)
    poly_modulus_degree = 16384 for 2-layer models  (more mult depth)
    Higher degree → more noise headroom but slower.
"""

if __name__ == "__main__":
    print(TENSEAL_UPGRADE_GUIDE)
    # Uncomment the line below and point to your CSV:
    # run_demo("breastcancer.csv")