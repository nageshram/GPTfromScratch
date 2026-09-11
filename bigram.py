import torch 
import torch.nn as nn
from torch.nn import functional as F

#source .venv/bin/activate
with open("input.txt") as f:
    data = f.read()

#print(data[:600])

# getting the size of the vocabulary
vocab_size= len(set(data))
#print(vocab_size)

# encoding the characters
stoi = { s :i for i,s in enumerate(sorted(set(data)))}

#decoding the characters 
itos = { i: s for i,s in enumerate(sorted(set(data)))}

#config
batch_size = 64 # how many independent sequences we process in parallel
block_size = 256  # the context length or maximum sequence length the model can see at once
n_embed = 384 # Embedding dimension (set to 64 to align with num_heads * head_size)
num_heads = 4
head_size = n_embed // num_heads
eval_interval = 500
eval_iters = 200
max_iters = 10000
learning_rate = 1e-3
dropout_rate = 0.2 # 20% of neurons will be shut 
n_layer = 6
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
#print(device)

torch.manual_seed(1337)

#encode the data
#encoder_data = [stoi[c] for c in data[:600]]
#print(encoder_data)
#print("".join(itos[i] for i in encoder_data))

#split the data into train and test 
percentage_train = 0.9*len(data)
train_data = data[:int(percentage_train)]
test_data = data[int(percentage_train):]

#encode the data :
train = [stoi[c] for c in train_data]
test = [stoi[c] for c in test_data]


def get_batch(split):
    data_to_use = train if split == 'train' else test

    ix = torch.randint(len(data_to_use) - block_size, (batch_size,)) # in order to slide it subtracted
    x = torch.tensor([data_to_use[i:i+block_size] for i in ix], dtype=torch.long) 
    y = torch.tensor([data_to_use[i+1: i+block_size+1] for i in ix], dtype=torch.long)
    return x.to(device), y.to(device)



#self Attention Block with head_ size = 16

class Head(nn.Module):

    def __init__(self,head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout_rate)

    
    def forward(self, x):
        """
        - Matrice rule no of cols of 1st matrice = no of rows in 2nd matrice  but here
          B,T,C hold by two matrices q and k so it doesnt match so we need to transpose k to (B,C,T) and multiply with q which is (B,T,C) so we get (B,T,T)
        
        - 8,8,64 @ 8,64,8 -> 8,8,8  # this is for one batch , for 4 batches it will be 4,8,8
         and that -sqrt(C) to control the variance ( spreadness of data from the mean ) the larger the variance the more likely the large values , so we divide by sqrt(head_size) to keep the variance under control during initialization
        
        - we mask the upper triangle of the matrix by setting the values to -inf 
          because we don't want the model to look at the future tokens , if it looks at the future tokens it won't generate or learn the next sequence properly
          
        """
        B,T,C = x.shape
        k = self.key(x)   # B,T,head_size
        q = self.query(x)  # B,T,head_size
       
        wei = (q @ k.transpose(-2,-1)) * (self.head_size**-0.5)  # B,T,T # transpose here because k is (B,T,head_size) and we want to multiply with q which is (B,T,head_size) so we need to transpose k to (B,head_size,T) 
        # [:T, :T] slices the max (block_size, block_size) tril matrix to match the current sequence length T (e.g., T=1 up to block_size during generation)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # upto T rows and T columns slices the sub matrice, (B, T, T) mask future positions with -inf so softmax turns them to 0 
        wei = F.softmax(wei, dim=-1) # B,T,T , we did softmax on the last dimension so that it sums up to 1 across columns
        wei = self.dropout(wei)

        # performing weighted aggregation of the values 
        v = self.value(x)  # B,T,head_size
        out = wei @ v # B,T,head_size (this is the output of the single head self attention)
        
        return out



class MultiHeadAttention(nn.Module):
    def __init__(self,num_heads,head_size):
        
        """
        multiple heads attention in parallel , concatenated and passed through linear projection layer
        """
        super().__init__()
        self.sa_heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed) # residual connections added after combining heads, so this linear layer will scale the combined output to the original embedding size    
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self,x):
        """
        pass single self attention heads in parallel
        concatenate the output of the heads
        pass through linear projection layer ( using self.proj() ) -> final output is of shape (B,T,C)
        Projection is nothing but passing the raw values through linear layer to mix the information from different heads
        """
        out = torch.cat([h(x) for h in self.sa_heads], dim = -1) # concatenation using last dimension
        out = self.proj(out) # projection
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    """
    helps to think individually after self attention ( like thoughts ),
    since the calculation comes from self attention if we do back propagation the grdaients will give us better learning 
    """
    def __init__(self,n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed,4*n_embed),   # first linear layer
            nn.ReLU(),
            nn.Linear(4*n_embed,n_embed), # second linear layer
            nn.Dropout(dropout_rate) #Dropout is used to prevent overfitting
        )

    def forward(self,x):
        return self.net(x)

class Block(nn.Module):
    """
    Combination of multi head self attention and feed forward with layer normalization and residual connection
    Transformer Block : Self Attention -> Layer Norm -> Feed Forward -> Layer Norm : communication Followed by Computation
    """
    def __init__(self,n_embed, num_heads):
        super().__init__()
        self.sa = MultiHeadAttention(num_heads=num_heads, head_size=n_embed//num_heads)
        self.ff = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed) # by paper that normalizes using the formula : 
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self,x):
        x = x + self.sa(self.ln1(x))  # LayerNorm before self attention : x + self.sa(x)
        x = x + self.ff(self.ln2(x)) # LayerNorm before feed forward : x + self.ff(x)
        return x


#define the model 
class BigramLanguageModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding = torch.nn.Embedding(vocab_size, n_embed)
        self.positional_embedding = torch.nn.Embedding(block_size, n_embed) # for each position in the block size will get the vectors representing that position
        self.blocks = nn.Sequential(*[Block(n_embed, num_heads=num_heads) for _ in range(n_layer)]) # multi head self attention
        self.ln_f = nn.LayerNorm(n_embed)  # final layer normalization
        self.lm_head = torch.nn.Linear(n_embed, vocab_size) # linear layer to map the output of the self attention to the vocabulary size 

       
    def forward(self,idx, target= None):
        B,T = idx.shape # B is batch size, T is the block size
        tok_emb = self.token_embedding(idx) # B,T,n_embed
        pos_emb = self.positional_embedding(torch.arange(T, device=idx.device)) # T,n_embed
        x = tok_emb + pos_emb # token_emb + positional_emb , shape is same (B,T,n_embed)   
        x = self.blocks(x) # output of multi head self attention (B,T,n_embed)
        x = self.ln_f(x) # final layer normalization
        logits = self.lm_head(x) # B,T,Vocab_size
        
        if target is None:
            return logits
        else:
            B,T,C = logits.shape # logits is B,T,Vocab_size
            logits = logits.view(B*T, C) # B*T, Vocab_size
            target = target.view(B*T) # B*T,
            loss = F.cross_entropy(logits, target)
            return logits, loss

    def generate(self, idx, max_new_tokens=100):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]# means last block_size tokens
            logits = self(idx_cond) # self(idx) calls the forward function
            logits = logits[:, -1, :] # taking last token as we are predicting the next token based on the previous ones 
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1) # appends the next token to the sequence 
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) # 1e-3

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# final evaluation
losses = estimate_loss()
print(f"step {max_iters}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

print("\n--- Generating text from trained model ---")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
out = model.generate(context, 100)
print("".join([itos[i] for i in out[0].tolist()]))

# Save model checkpoint and metadata to pickle file
checkpoint = {
    'model_state_dict': model.state_dict(),
    'config': {
        'vocab_size': vocab_size,
        'n_embed': n_embed,
        'block_size': block_size,
        'num_heads': num_heads,
        'head_size': head_size,
        'n_layer': n_layer,
        'dropout_rate': dropout_rate,
    },
    'stoi': stoi,
    'itos': itos
}
torch.save(checkpoint, 'model.pkl')
print("\nModel saved successfully to model.pkl")


        



















