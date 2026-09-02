import os
import sys
from typing import Optional
from src.tokenizer import get_global_tokenizer
from src.model import CausalTransformerLM, HAS_TORCH

if HAS_TORCH:
    import torch

WEIGHTS_PATH = os.path.join("submission_artifacts", "checkpoints", "mini_model_weights.pt")

QA_PAIRS = [
    ("Hello?", "Hello! How can I help you today?"),
    ("What is the capital of France?", "The capital of France is Paris."),
    ("What is V5?", "V5 is the Training Data Execution System for LLMs."),
    ("Who created this?", "Sathvik created this V5 Data Execution Engine."),
]


def train_mini_model(model: CausalTransformerLM, num_epochs: int = 150):
    """
    Trains CausalTransformerLM on Q&A dataset for 150 fast AdamW steps.
    """
    if not HAS_TORCH or not hasattr(model._model, "compute_loss_tensor"):
        return

    tokenizer = get_global_tokenizer()
    inner_model = model._model
    inner_model.train()
    optimizer = torch.optim.AdamW(inner_model.parameters(), lr=5e-3)

    print(f"\n[TRAINING] Training Causal Transformer LM on Q&A dataset ({num_epochs} steps)...")

    training_samples = []
    for q, a in QA_PAIRS:
        text = f"{q} {a}"
        tokens = tokenizer.encode(text, add_eod=True)
        loss_mask = [1.0] * len(tokens)
        training_samples.append((tokens, tokens, loss_mask))

    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        for input_ids, labels, loss_mask in training_samples:
            optimizer.zero_grad()
            loss = inner_model.compute_loss_tensor(input_ids, labels, loss_mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 50 == 0 or epoch == num_epochs:
            print(f"  Step {epoch:03d}/{num_epochs} — Loss: {total_loss / len(training_samples):.4f}")

    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    torch.save(inner_model.state_dict(), WEIGHTS_PATH)
    print(f"[SUCCESS] Trained model weights saved to {WEIGHTS_PATH}!\n")


def get_or_load_model(force_retrain: bool = False) -> CausalTransformerLM:
    model = CausalTransformerLM(vocab_size=512, hidden_dim=128, num_layers=2)
    if HAS_TORCH and hasattr(model._model, "load_state_dict"):
        if force_retrain or not os.path.exists(WEIGHTS_PATH):
            train_mini_model(model)
        else:
            model._model.load_state_dict(torch.load(WEIGHTS_PATH))
    return model


def talk_to_model(prompt: str, max_new_tokens: int = 35, model: Optional[CausalTransformerLM] = None) -> str:
    """
    Encodes prompt using BPE tokenizer, queries the Causal Transformer LM,
    and decodes generated token response into text.
    """
    if model is None:
        model = get_or_load_model()

    tokenizer = get_global_tokenizer()
    input_tokens = tokenizer.encode(prompt)
    print(f"[USER PROMPT]: {prompt}")

    # Autoregressive generation with temperature=0.1 (greedy prediction for trained Q&A)
    output_tokens = model.generate(input_tokens, max_new_tokens=max_new_tokens, temperature=0.1)
    response_text = tokenizer.decode(output_tokens)

    # Strip prompt prefix from output if present
    if response_text.startswith(prompt):
        answer_text = response_text[len(prompt):].strip()
    else:
        answer_text = response_text.strip()

    print(f"[MODEL RESPONSE]: {answer_text}\n")
    return answer_text


def main():
    print("=" * 65)
    print("  V5 Training Data Execution System — Trained Model CLI  ")
    print("=" * 65)

    force_retrain = "--train" in sys.argv
    model = get_or_load_model(force_retrain=force_retrain)

    # If prompt passed via CLI arguments
    user_args = [arg for arg in sys.argv[1:] if arg != "--train"]
    if user_args:
        prompt = " ".join(user_args)
        talk_to_model(prompt, model=model)
    else:
        print("Demonstrating trained model responses across Q&A prompts:\n")
        for q, _ in QA_PAIRS:
            talk_to_model(q, model=model)


if __name__ == "__main__":
    main()
