# GPT Model from Scratch

This project implements a basic GPT (Generative Pre-trained Transformer) model from scratch using PyTorch. The project includes training the model on dummy data and generating text using the trained model.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Files](#files)
- [Training the Model](#training-the-model)
- [Generating Text](#generating-text)
- [License](#license)

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/ljmullings/HomemadeGPT.git
   cd HomemadeGPT

2. **Create and activate a Conda environment:**
  ```bash
  conda create -n gpt-env python=3.8
  conda activate gpt-env
```
3. **Install the required packages:**

  ```bash
      pip install torch transformers
  ```
## Usage
### Files
- gpt.py: Defines the GPT model architecture.
- positional_encoding.py: Defines the positional encoding used in the GPT model.
- transformer_decoder_layer.py: Defines the transformer decoder layer used in the GPT model.
- self_attention.py: Defines the self-attention mechanism used in the transformer decoder layer.
- feed_forward.py: Defines the feed-forward neural network used in the transformer decoder layer.
- train.py: Script to train the GPT model.
- generate_text.py: Script to generate text using the trained GPT model.

### Training the Model
1. **Prepare the dummy data:**
Ensure your train.py script includes a function to generate or load dummy training data.

2. **Train the model:**
```bash
python train.py
```
This will train the model and save the trained model checkpoint to gpt_model.pth.

### Generating Text
1. **Generate text using the trained model:**

```bash
python generate_text.py
```
This script will load the trained model and generate text starting from a specified start token.

### Files
#### gpt.py
Defines the GPT model architecture. It includes word embedding, positional encoding, and multiple transformer decoder layers.

#### positional_encoding.py
Defines the positional encoding class used in the GPT model to provide positional information to the token embeddings.

#### transformer_decoder_layer.py
Defines the transformer decoder layer used in the GPT model.

#### self_attention.py
Defines the self-attention mechanism used in the transformer decoder layer.

#### feed_forward.py
Defines the feed-forward neural network used in the transformer decoder layer.

#### train.py
Script to train the GPT model using dummy data. It saves the model checkpoint to gpt_model.pth after training.

#### generate_text.py
Script to generate text using the trained GPT model. It starts with a specified start token and generates a sequence of tokens.

### Example Output
Example output when generating text:

```
Script started.
Using device: cuda
Loading model...
Model loaded and set to eval mode.
Tokenizer loaded.
Starting text generation...
Starting generation with start token: 0
Step 1: Generated token 123
Step 2: Generated token 456
...
Generated Text: "The quick brown fox jumps over the lazy dog."
```
## License
