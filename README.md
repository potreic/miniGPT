# miniGPT: Your Own Transformer from Scratch with NumPy! 🧠

Welcome to `miniGPT`! This project is a from-scratch implementation of a decoder-only Transformer model (like GPT) using nothing but **NumPy**. It's designed as a learning tool to demystify the magic behind modern language models.

Instead of relying on high-level frameworks like TensorFlow or PyTorch, we build every component—from token embeddings to the attention mechanism—by hand. This way, you can see the nuts and bolts of how a language model *really* works.

---

## The Building Blocks of miniGPT 🧱

Our model is built like a set of Lego blocks. Each class is a modular component with a specific job. Let's break them down!

### `Embedding` and `PositionalEncoding`
* **Token Embedding:** Think of this as the model's dictionary. It takes a token's ID number (e.g., `52`) and converts it into a meaningful vector of numbers (`[0.12, -0.45, ...]`). This vector, or "embedding," captures the token's semantic meaning.
* **Positional Encoding:** A sentence's meaning depends on word order ("the cat sat on the mat" is different from "the mat sat on the cat"). This module injects information about a token's position. It's like giving each word a GPS coordinate in the sequence.

### A Key Design Choice: Learned Positional Encoding

Instead of using a fixed mathematical formula (like sine and cosine waves), this model uses **learned positional encoding**. Here's the justification:

> **Flexibility and Simplicity!** 🎯 The model learns the *optimal* way to represent positions directly from the training data, tailoring itself to the specific patterns in the text. It's not locked into a one-size-fits-all formula. Plus, the implementation is incredibly simple—it's just another `Embedding` layer where the "vocabulary" is the list of positions (0, 1, 2, ...). The main trade-off is that it can't easily generalize to sequences longer than it was trained on, but for a fixed context length, it's highly effective.

---

### `MultiHeadAttention`: The Model's Brain 🧠

This is the core of the Transformer. The "Attention" mechanism allows the model to weigh the importance of different tokens in the sequence when producing the next one.

Imagine you're trying to understand a word in a sentence. You pay more "attention" to related words. That's what this module does!

1.  **Q, K, V (Query, Key, Value):** For each token, we create three vectors:
    * **Query (Q):** My current state, asking, "Who should I pay attention to?"
    * **Key (K):** A token's "label" or "ID card," saying, "This is what I am."
    * **Value (V):** The actual content or meaning of that token.

2.  **The Process:** The model takes the **Query** from the current token and compares it with the **Key** of every other token in the sequence. The better the match, the higher the score. These scores are then used to create a weighted sum of all the **Values**.

3.  **Multi-Head:** Why just do this once? "Multi-Head" attention means we have several "teams of experts" (heads) doing this in parallel. One head might focus on grammatical relationships, while another focuses on semantic meaning. The results are then combined, giving a much richer understanding of the text.

4.  **Causal Masking (No Spoilers!):** For a language model that predicts the future, it would be cheating to see the answer! The causal mask is like putting blinders on the model. It ensures that when predicting the token at position `t`, the model can only pay attention to tokens from `0` to `t`. All future tokens are hidden.

---

### `FeedForward`, `LayerNorm`, and `Residual Connections`
* **Feed-Forward Network (FFN):** After the attention mechanism gathers information from other tokens, the FFN acts as the "thinking" or "processing" unit. Each token's vector is passed through this small neural network to process the information gathered.

* **Residual Connections (`+`):** This is a simple but powerful trick: we add the input of a layer to its output (`X + Attention(X)`). This "shortcut" helps prevent the model from "forgetting" the original information as it passes through many layers. It's a key factor in training deep networks successfully.

* **Layer Normalization:** Think of this as a volume stabilizer. It keeps the numbers flowing through the network in a nice, stable range, which makes the training process much smoother. In our model, we use a **pre-norm** architecture: we normalize the input *before* it enters the attention or FFN layers.

---

### `TransformerBlock` and the Final `Linear` Layer
* **`TransformerBlock`:** This simply packages one `MultiHeadAttention` layer and one `FeedForward` layer together, along with their residual connections and layer normalizations. Our `miniGPT` is just a stack of these blocks!

* **Final Output Layer:** After the final token vectors emerge from the last `TransformerBlock`, a `Linear` layer projects this high-dimensional vector into a huge vector the size of our vocabulary. This final vector contains the raw scores (logits) for every possible next token. A **softmax** function is then applied to these scores to turn them into probabilities!

---

## Tokenizer Showdown: Word vs. BPE

The notebook includes two ways to turn text into tokens:

* **`WordTokenizer`:** A simple, intuitive tokenizer that splits text by words and punctuation. It's easy to understand but can struggle with rare words and results in a large vocabulary.
* **`BPETokenizer` (Byte-Pair Encoding):** A smarter tokenizer. It starts with individual characters and iteratively merges the most frequent pairs of tokens. It learns sub-word units (like `ing` or `ly`), which allows it to handle any word, even ones it's never seen, and keeps the vocabulary size manageable.

---

## How to Use This Notebook 🚀

1.  **Prepare the Data:** The notebook starts by reading a text file (e.g., `shakespeare.txt`), building a vocabulary with a tokenizer, and splitting the data into training and validation sets.

2.  **Define the Model:** Configure the hyperparameters for your `miniGPT` instance.
    ```python
    model = miniGPT(
        vocab_size=vocab_size,
        max_len=block_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff
    )
    ```

3.  **Train the Model:** The training loop is a simple implementation of Stochastic Gradient Descent (SGD).
    ```python
    for it in range(num_iters):
        # Get a batch of data
        x, y = get_batch(train_data, block_size, batch_size)

        # Forward pass to get logits and loss
        logits, loss = model.forward(x, targets=y)

        # Backward pass to calculate gradients
        model.backward()

        # Update model weights
        for name, param in model.params().items():
            grad = model.grads()[name]
            param -= learning_rate * grad
    ```

4.  **Generate Text:** Once trained, you can give the model a starting prompt and let it generate new text!
    ```python
    context = np.array([[tokenizer.encode("my dear")[0]]])
    generated_ids = model.generate(context, max_new_tokens=50)
    print(tokenizer.decode(generated_ids[0]))
    ```

---

## Checking Our Work 🧐

The notebook includes a simple unit test cell to verify that the core components are behaving as expected. These tests check:
1.  **Tensor Dimensions:** Ensures the model's output shape is correct.
2.  **Softmax Properties:** Confirms the output probabilities sum to 1.
3.  **Causal Masking:** Verifies that the model cannot "cheat" by looking at future tokens.

These checks are crucial for catching bugs and building confidence in the model's implementation.
