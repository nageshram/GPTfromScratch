import torch
from bigram import BigramLanguageModel, device

# Load model checkpoint from pickle file
checkpoint_path = 'model.pkl'
print(f"Loading checkpoint from '{checkpoint_path}' on device '{device}'...")
checkpoint = torch.load(checkpoint_path, map_location=device)

config = checkpoint['config']
stoi = checkpoint['stoi']
itos = checkpoint['itos']

vocab_size = config['vocab_size']

# Instantiate model architecture and load saved weights
model = BigramLanguageModel(vocab_size).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("Model successfully loaded and ready for inference!\n")

# Set seed for reproducible generation
torch.manual_seed(1337)

context = torch.zeros((1, 1), dtype=torch.long, device=device)
out = model.generate(context, max_new_tokens=500)

print("--- Generated Text (Reproduced) ---")
print("".join([itos[i] for i in out[0].tolist()]))
