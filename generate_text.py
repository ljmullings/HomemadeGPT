import torch
from transformers import GPT2Tokenizer
from gpt import GPT

# Set the device to CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
embed_size = 512
heads = 8
forward_expansion = 4
num_layers = 6
max_len = 100
dropout = 0.3
vocab_size = 50257

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

model = GPT(embed_size, heads, forward_expansion, num_layers, vocab_size, max_len, dropout).to(device)

# Load the model checkpoint
checkpoint_path = '.\models\your_modelv1.pth'
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def generate_text(model, tokenizer, start_text, max_length=100, temperature=0.5, top_k=50):
    model.eval()
    tokens = tokenizer.encode(start_text, return_tensors='pt').to(device)
    generated = tokens

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(generated, None)
            next_token_logits = outputs[:, -1, :] / temperature

            # Apply top-k sampling
            if top_k > 0:
                top_k_values, top_k_indices = torch.topk(next_token_logits, top_k)
                next_token_probs = torch.softmax(top_k_values, dim=-1)
                next_token = top_k_indices.gather(-1, torch.multinomial(next_token_probs, 1))
            else:
                next_token_probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(next_token_probs, 1)

            # Ensure next_token is a 2D tensor with shape (batch_size, 1)
            next_token = next_token.squeeze(-1)

            # Concatenate the next token to the generated sequence
            generated = torch.cat((generated, next_token.unsqueeze(-1)), dim=1)

            if next_token[0].item() == tokenizer.eos_token_id:
                break

    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return output_text

if __name__ == "__main__":
    start_text = "<Start Text>"
    print("Generating text...")
    generated_text = generate_text(model, tokenizer, start_text, max_length=50)  # Increase max_length for more output
    print("Generated text:", generated_text)
