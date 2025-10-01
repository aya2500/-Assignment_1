# 📄 Assignment 1: Debugging Transformer Models with PyCharm & WSL  

## 📌 Overview  
This assignment focuses on **debugging a Transformer model** using **PyCharm** with **WSL (Windows Subsystem for Linux)**.  
Instead of print statements, the **PyCharm debugger** was used to trace tensors at different stages of the Transformer pipeline.  

The model was tested with an example from the **literature domain**:  
- **Input sequence (5 tokens, IDs):** `[[511, 723, 845, 932, 678]]`  
- **Target sequence (5 tokens, IDs):** `[[812, 459, 511, 390, 275]]`  

---

## ⚙️ Environment Setup  
- **IDE:** PyCharm  
- **Interpreter:** WSL (Python environment)  
- **Python version:** 3.9+ (recommended)  
- **Dependencies:** listed inside `requirements.txt` (included in the project ZIP).  

---

## ▶️ How to Run the Code  

### **1. Clone the repository**  
```bash
git clone https://github.com/aya2500/assignment1-transformer-debug.git
cd assignment1-transformer-debug
```

### **2. Extract project files**  
The project files (code, documentation, requirements) are compressed in a ZIP archive.  
```bash
unzip assignment1.zip
cd assignment1
```

### **3. Install dependencies**  
```bash
pip install -r requirements.txt
```

### **4. Open with PyCharm**  
- Configure **WSL Python interpreter** inside PyCharm:  
  - `File → Settings → Project → Python Interpreter → Add Interpreter → WSL`  
- Select the created environment (with installed packages).  

### **5. Run the code**  
- Run the main script:  
  ```bash
  python transformer_debug.py
  ```
- Or run directly from **PyCharm** with the debugger attached.  

---

## 🔍 Debugging Snapshots  
During debugging, the following stages were inspected:  

1. **Raw input tokens (IDs):** `(1, 5)`  
2. **Target tokens (IDs):** `(1, 5)`  
3. **Embedding lookup & positional encoding:** `(1, 5, 128)`  
4. **Self-attention (Q, K, V):** `(1, 5, 128)`  
5. **Attention scores (before & after softmax):** `(1, 4, 5, 5)`  
6. **Multi-head attention output:** `(1, 5, 128)`  
7. **Residual connection & LayerNorm:** `(1, 5, 128)`  
8. **Feed Forward (FFN):** `(1, 5, 128) → (1, 5, 512) → (1, 5, 128)`  
9. **Decoder masked self-attention with causal masking**  
10. **Cross-attention with encoder output**  
11. **Final decoder output & vocabulary logits:** `(1, 5, 10000)`  

---

## ❓ Guiding Questions & Answers  

- **Q1:** What do dimensions represent at each stage?  
  **A1:**  
  - Embedding → `(batch, seq_len, d_model)`  
  - Attention → `(batch, heads, seq_len, d_head)`  
  - Feed Forward → `(batch, seq_len, d_ff)`  
  - Output → `(batch, seq_len, vocab_size)`  

- **Q2:** Why do Q, K, V have the same shape?  
  **A2:** To ensure valid dot products. They are split into heads to capture diverse relationships.  

- **Q3:** Why are attention matrices square?  
  **A3:** Each token compares with every other token in the sequence.  

- **Q4:** Why is masking necessary in the decoder?  
  **A4:** To prevent looking ahead at future tokens, ensuring causal prediction.  

- **Q5:** Role of residuals & normalization?  
  **A5:** Keep consistent tensor shapes and stable training dynamics.  

- **Q6:** Why keep embedding dimension constant?  
  **A6:** To ensure residual connections work without shape mismatch.  

- **Q7:** How does final projection connect to logits?  
  **A7:** Linear layer maps `d_model → vocab_size`, producing token probabilities.  

---

## ✅ Conclusion  
This assignment demonstrates step-by-step debugging of a Transformer model inside **PyCharm with WSL**, with all logs and explanations included in the provided documentation.  
