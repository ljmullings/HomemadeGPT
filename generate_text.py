import torch
from transformers import GPT2Tokenizer
from gpt import GPT
import os

def main():
    print("Script started.")
    
    # check if GPU is available and use it if possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ensure model path exists
    model_path = 'gpt_model.pth'
    if not os.path.exists(model_path):
        print(f"Model path {model_path} does not exist.")
        return

    print("Loading model...")
    
    # load the trained model
    model = GPT(embed_size=512, heads=8, forward_expansion=4, num_layers=6, vocab_size=10000, max_len=100, dropout=0.1)
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    model.to(device)
    model.eval()
    print("Model loaded and set to eval mode.")

    try:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    print("Tokenizer loaded.")

    # function to generate text
    def generate_text(model, start_token, max_length=50):
        model.eval()
        tokens = torch.tensor([start_token]).unsqueeze(0).to(device)
        print(f"Starting generation with start token: {start_token}")

        for i in range(max_length):
            mask = None
            print(f"Tokens shape before model: {tokens.shape}")
            with torch.no_grad():
                if tokens.max().item() >= model.fc_out.out_features or tokens.min().item() < 0:
                    raise ValueError(f"Token index out of bounds: {tokens}")
                outputs = model(tokens, mask)
                next_token_logits = outputs[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                tokens = torch.cat((tokens, next_token), dim=1)  # Append next token
                print(f"Step {i + 1}: Generated token {next_token.item()}")

        return tokens.squeeze().tolist()

    start_token = 0 

    # generate sample output
    print("Starting text generation...")
    sample_output = generate_text(model, start_token)
    print("Text generation complete.")

    # convert token IDs to text
    generated_text = tokenizer.decode(sample_output, skip_special_tokens=True)
    print("Generated Text:", generated_text)

if __name__ == "__main__":
    # set for better debugging
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    main()
