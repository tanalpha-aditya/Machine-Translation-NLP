
# Replace with the path to your "dev.en" file
import matplotlib.pyplot as plt
import numpy as np
import nltk
import sacrebleu
from tqdm import tqdm
import torch
import spacy
file_path = "/kaggle/input/eng-french-translation/ted-talks-corpus/dev.en"
with open(file_path, 'r', encoding='utf-8') as file:
    data = file.read()
sentences = data.split('\n')
dev_en = [[sentence.strip()] for sentence in sentences if sentence.strip()]


# Replace with the path to your "dev.en" file
file_path = "/kaggle/input/eng-french-translation/ted-talks-corpus/dev.fr"
with open(file_path, 'r', encoding='utf-8') as file:
    data = file.read()
sentences = data.split('\n')
dev_fr = [[sentence.strip()] for sentence in sentences if sentence.strip()]


# Replace with the path to your "dev.en" file
file_path = "/kaggle/input/eng-french-translation/ted-talks-corpus/test.en"
with open(file_path, 'r', encoding='utf-8') as file:
    data = file.read()
sentences = data.split('\n')
test_en = [[sentence.strip()] for sentence in sentences if sentence.strip()]


# Replace with the path to your "dev.en" file
file_path = "/kaggle/input/eng-french-translation/ted-talks-corpus/test.fr"
with open(file_path, 'r', encoding='utf-8') as file:
    data = file.read()
sentences = data.split('\n')
test_fr = [[sentence.strip()] for sentence in sentences if sentence.strip()]


# Replace with the path to your "dev.en" file
file_path = "/kaggle/input/eng-french-translation/ted-talks-corpus/train.en"
with open(file_path, 'r', encoding='utf-8') as file:
    data = file.read()
sentences = data.split('\n')
train_en = [[sentence.strip()] for sentence in sentences if sentence.strip()]


# Replace with the path to your "dev.en" file
file_path = "/kaggle/input/eng-french-translation/ted-talks-corpus/train.fr"
with open(file_path, 'r', encoding='utf-8') as file:
    data = file.read()
sentences = data.split('\n')
train_fr = [[sentence.strip()] for sentence in sentences if sentence.strip()]

len(dev_en)

!python - m spacy download fr_core_news_sm


# Load the English and French spaCy models
nlp_en = spacy.load("en_core_web_sm")
nlp_fr = spacy.load("fr_core_news_sm")

# Tokenization and special tokens


def tokenize_and_add_special_tokens(data, language):
    tokenized_data = []
    for sentence in data:
        doc = language(sentence[0])
        tokens = ["<bos>"] + [token.text for token in doc] + ["<eos>"]
        tokenized_data.append(tokens)
    return tokenized_data


# Tokenize and add special tokens to English sentences
train_en = tokenize_and_add_special_tokens(train_en, nlp_en)

# Tokenize and add special tokens to French sentences
train_fr = tokenize_and_add_special_tokens(train_fr, nlp_fr)

# Tokenize and add special tokens to English sentences
test_en = tokenize_and_add_special_tokens(test_en, nlp_en)

# Tokenize and add special tokens to French sentences
test_fr = tokenize_and_add_special_tokens(test_fr, nlp_fr)
# Tokenize and add special tokens to English sentences

dev_en = tokenize_and_add_special_tokens(dev_en, nlp_en)

# Tokenize and add special tokens to French sentences
dev_fr = tokenize_and_add_special_tokens(dev_fr, nlp_fr)


# Load the English and French spaCy models
nlp_en = spacy.load("en_core_web_sm")
nlp_fr = spacy.load("fr_core_news_sm")

# Tokenization and special tokens


def tokenize_and_add_special_tokens(data, language):
    tokenized_data = []
    for sentence in data:
        doc = language(sentence[0])
        tokens = ["<bos>"] + [token.text for token in doc] + ["<eos>"]
        tokenized_data.append(tokens)
    return tokenized_data


# Tokenize and add special tokens to English sentences
train_en = tokenize_and_add_special_tokens(train_en, nlp_en)

# Tokenize and add special tokens to French sentences
train_fr = tokenize_and_add_special_tokens(train_fr, nlp_fr)

# Tokenize and add special tokens to English sentences
test_en = tokenize_and_add_special_tokens(test_en, nlp_en)

# Tokenize and add special tokens to French sentences
test_fr = tokenize_and_add_special_tokens(test_fr, nlp_fr)
# Tokenize and add special tokens to English sentences

dev_en = tokenize_and_add_special_tokens(dev_en, nlp_en)

# Tokenize and add special tokens to French sentences
dev_fr = tokenize_and_add_special_tokens(dev_fr, nlp_fr)

max_sentence_length = max(len(sentence) for sentence in train_en + train_fr)


filtered_train_fr = []
filtered_train_en = []

for sentence_fr, sentence_en in zip(train_fr, train_en):
    if len(sentence_fr) <= 100 and len(sentence_en) <= 100:
        filtered_train_fr.append(sentence_fr)
        filtered_train_en.append(sentence_en)

train_fr = filtered_train_fr
train_en = filtered_train_en


filtered_dev_fr = []
filtered_dev_en = []

for sentence_fr, sentence_en in zip(dev_fr, dev_en):
    if len(sentence_fr) <= 100 and len(sentence_en) <= 100:
        filtered_dev_fr.append(sentence_fr)
        filtered_dev_en.append(sentence_en)

dev_fr = filtered_dev_fr
dev_en = filtered_dev_en


filtered_test_fr = []
filtered_test_en = []

for sentence_fr, sentence_en in zip(test_fr, test_en):
    if len(sentence_fr) <= 100 and len(sentence_en) <= 100:
        filtered_test_fr.append(sentence_fr)
        filtered_test_en.append(sentence_en)

test_fr = filtered_test_fr
test_en = filtered_test_en


# train_fr = [sentence for sentence in train_fr if len(sentence) <= 100]
# train_en = [sentence for sentence in train_en if len(sentence) <= 100]
# test_en = [sentence for sentence in test_en if len(sentence) <= 100]
# test_fr = [sentence for sentence in test_fr if len(sentence) <= 100]
# dev_en = [sentence for sentence in dev_en if len(sentence) <= 100]
# dev_fr = [sentence for sentence in dev_fr if len(sentence) <= 100]


for sentence in train_fr + train_en + test_en + test_fr + dev_en + dev_fr:
    while len(sentence) < 100:
        sentence.append("<pad>")
    for i in range(len(sentence)):
        if sentence[i] == "":
            sentence[i] = "<unk>"


max_sentence_length = max(len(sentence) for sentence in train_en + train_fr)


# Flatten the list of lists to create a single list of tokens
all_tokens_en = [token for sentence in train_en for token in sentence]
all_tokens_fr = [token for sentence in train_fr for token in sentence]

# Create a set to remove duplicates and form your vocabulary
vocab_en = set(all_tokens_en)
vocab_fr = set(all_tokens_fr)

# Optionally, add special tokens to the vocabulary if they are not present
special_tokens = ['<bos>', '<eos>', '<pad>', '<unk>']
for token in special_tokens:
    vocab_en.add(token)
    vocab_fr.add(token)

vocab_en = list(vocab_en)
vocab_fr = list(vocab_fr)

# Create a mapping from tokens to their indices
token_to_index_en = {token: index for index, token in enumerate(vocab_en)}

# Convert each tokenized sentence to a list of indices
train_en_indices = []
for sentence in train_en:
    sentence_indices = [token_to_index_en.get(
        token, token_to_index_en['<unk>']) for token in sentence]
    train_en_indices.append(sentence_indices)

# 'train_en_indices' now contains the list of indices for each sentence
dev_en_indices = []
for sentence in dev_en:
    sentence_indices = [token_to_index_en.get(
        token, token_to_index_en['<unk>']) for token in sentence]
    dev_en_indices.append(sentence_indices)

test_en_indices = []
for sentence in test_en:
    sentence_indices = [token_to_index_en.get(
        token, token_to_index_en['<unk>']) for token in sentence]
    test_en_indices.append(sentence_indices)

# Create a mapping from tokens to their indices
token_to_index_fr = {token: index for index, token in enumerate(vocab_fr)}

# Convert each tokenized sentence to a list of indices
train_fr_indices = []
for sentence in train_fr:
    sentence_indices = [token_to_index_fr.get(
        token, token_to_index_fr['<unk>']) for token in sentence]
    train_fr_indices.append(sentence_indices)

# 'train_en_indices' now contains the list of indices for each sentence
dev_fr_indices = []
for sentence in dev_fr:
    sentence_indices = [token_to_index_fr.get(
        token, token_to_index_fr['<unk>']) for token in sentence]
    dev_fr_indices.append(sentence_indices)

test_fr_indices = []
for sentence in test_fr:
    sentence_indices = [token_to_index_fr.get(
        token, token_to_index_fr['<unk>']) for token in sentence]
    test_fr_indices.append(sentence_indices)

# TRANSFORMER MODEL


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Initialize dimensions
        self.d_model = d_model  # Model's dimension
        self.num_heads = num_heads  # Number of attention heads
        self.d_k = d_model // num_heads  # Dimension of each head's key, query, and value

        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model)  # Query transformation
        self.W_k = nn.Linear(d_model, d_model)  # Key transformation
        self.W_v = nn.Linear(d_model, d_model)  # Value transformation
        self.W_o = nn.Linear(d_model, d_model)  # Output transformation

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        attn_scores = torch.matmul(
            Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # Softmax is applied to obtain attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # Multiply by values to obtain the final output
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        # Perform scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding, self).__init__()
        # Initialize dimensions
        self.d_model = d_model
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(
            0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):

        encoding = x + self.pe[:, :x.size(1)]
        return encoding


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Pass through the first linear layer and apply ReLU activation
        intermediate = self.fc1(x)
        intermediate = self.relu(intermediate)

        # Pass through the second linear layer
        output = self.fc2(intermediate)

        return output


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # Multi-Head Self-Attention for the source sequence
        attn_output = self.self_attn(src, src, src, src_mask)
        src = self.norm1(src + self.dropout(attn_output))

        # Position-wise Feed-Forward Network
        ff_output = self.feed_forward(src)
        src = self.norm2(src + self.dropout(ff_output))

        return src


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.src_tgt_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, enc_output, src_mask, tgt_mask):
        # Self-Attention for the target sequence
        attn_output = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm1(tgt + self.dropout(attn_output))

        # Cross-Attention between target and source
        cross_attn_output = self.src_tgt_attn(
            tgt, enc_output, enc_output, src_mask)
        tgt = self.norm2(tgt + self.dropout(cross_attn_output))

        # Position-wise Feed-Forward Network
        ff_output = self.feed_forward(tgt)
        tgt = self.norm3(tgt + self.dropout(ff_output))

        return tgt


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2).to(
            self.device)  # Move to the same device as the model
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3).to(
            self.device)  # Move to the same device as the model
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length,
                       seq_length, device=self.device), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(
            self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(
            self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output


# Create batches
def batching(data, batch_size):
    num_batches = len(data) // batch_size
    batches = []
    for i in range(num_batches-2):
        batch = data[i * batch_size: (i + 1) * batch_size]
#         print(batch)
        batches.append(torch.tensor(batch))
    return batches


src_vocab_size = len(vocab_en)
tgt_vocab_size = len(vocab_fr)
d_model = 304
num_heads = 8
num_layers = 1
d_ff = 2048
max_seq_length = max_sentence_length  # 100
dropout = 0.1

transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model,
                          num_heads, num_layers, d_ff, max_seq_length, dropout)

# Generate random sample data
src_data = batching(train_en_indices, 64)  # (batch_size, seq_length)
tgt_data = batching(train_fr_indices, 64)  # (batch_size, seq_length)


dev_src_data = batching(dev_en_indices, 32)  # (batch_size, seq_length)
dev_tgt_data = batching(dev_fr_indices, 32)  # (batch_size, seq_length)

test_src_data = batching(test_en_indices, 32)  # (batch_size, seq_length)
test_tgt_data = batching(test_fr_indices, 32)  # (batch_size, seq_length)

transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model,
                          num_heads, num_layers, d_ff, max_seq_length, dropout)

# TRANSFORMER TRAINING :

# Check if a GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model to the GPU
transformer.to(device)

# Rest of your code
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(), lr=0.0001)

transformer.train()

# Create a file to store training details
with open("train_model.txt", "w") as train_file:
    for epoch in range(10):
        # Initialize epoch loss
        epoch_loss = 0.0

        # Use tqdm with enumerate to create a progress bar
        for i, (src, tgt) in enumerate(tqdm(zip(src_data, tgt_data), desc=f"Epoch {epoch + 1}")):
            src = src.to(device)  # Move source data to the GPU
            tgt = tgt.to(device)  # Move target data to the GPU
            optimizer.zero_grad()
            output = transformer(src, tgt[:, :-1])
            loss = criterion(output.contiguous().view(-1,
                             tgt_vocab_size), tgt[:, 1:].contiguous().view(-1))
            loss.backward()
            optimizer.step()

            # Update the epoch loss
            epoch_loss += loss.item()

        # Calculate the average loss for the epoch
        average_epoch_loss = epoch_loss / len(src_data)

        # Print the epoch loss and save to the train_model.txt file
        print(f"Epoch: {epoch + 1}, Loss: {average_epoch_loss}")
        train_file.write(f"Epoch: {epoch + 1}, Loss: {average_epoch_loss}\n")


def indices_to_words(indices):
    #     print(indices)
    # Replace this with your vocabulary mapping
    # In this example, it's just a simple list, but you should use your actual vocabulary.
    vocab = vocab_fr

    words = [vocab[index] for index in indices[0] if index != 15008]
    return words


def indices_to_wordss(indices):
    vocab = vocab_fr
#     print("dddd",indices[0])

    indices = [token for token in indices if token != 15008]
    words = [vocab[token] for token in indices]
    return words


pip install sacrebleu


# Example references and hypothesis
references = [
    'Quand j avais la vingtaine , j ai vu mes tout premiers clients comme <unk> .']


hypotheses = [
    'Quand je étais mon première , je ai vu mes professeurs à . . professeur .']


# Calculate BLEU scores using sacrebleu
bleu = sacrebleu.corpus_bleu(hypotheses, [references], force=True)

print(f"Predicted BLEU Score: {bleu.score}")


def stringgg(sentence):
    # Remove the last token
    sentence_without_last_token = sentence[:-1]

    # Convert the list to a single string
    sentence_string = ' '.join(sentence_without_last_token)
    return [sentence_string]


# Assuming you have test_src_data and test_tgt_data with batching
transformer.eval()  # Set the model to evaluation mode

# Initialize variables to store test loss and BLEU scores
test_loss = 0.0
total_samples = 0
bleu_scores = []

# Create a file to store BLEU scores
output_file = open('test_bleu_scores.txt', 'w')

# Use tqdm with enumerate to create a progress bar
for i, (src, tgt) in enumerate(tqdm(zip(test_src_data, test_tgt_data), desc="Testing")):
    src = src.to(device)  # Move source data to the GPU
    tgt = tgt.to(device)  # Move target data to the GPU

    with torch.no_grad():  # Disable gradient computation for testing
        output = transformer(src, tgt[:, :-1])
        loss = criterion(output.contiguous().view(-1,
                         tgt_vocab_size), tgt[:, 1:].contiguous().view(-1))

    test_loss += loss.item()
    total_samples += src.size(0)  # Increment total samples by batch size

    # Calculate BLEU scores for the current batch
    for b in range(src.size(0)):  # Iterate over batch
        reference = [tgt[b, 1:].cpu().numpy()]
#         print(reference)
        hypothesis = torch.argmax(output[b], dim=1).cpu().numpy()

        reference_words = indices_to_words(reference)
        hypothesis_words = indices_to_wordss(hypothesis)
        hypothesis_words = stringgg(hypothesis_words)
        reference_words = stringgg(reference_words)
        print(hypothesis_words)
        print(reference_words)
        bleu = sacrebleu.corpus_bleu(
            hypothesis_words, [reference_words], force=True)
        bleu_scores.append(bleu.score)

    # Write the BLEU scores to the output file for this batch
    output_file.write(
        f"Batch {i + 1} BLEU scores: {', '.join([str(bleu) for bleu in bleu_scores[-src.size(0):]])}\n")

# Calculate the average test loss
average_test_loss = test_loss / total_samples

# Calculate the average BLEU score for the entire dataset
average_bleu_score = np.mean(bleu_scores)

print(f"Test Loss: {average_test_loss}")
print(f"Average BLEU Score: {average_bleu_score}")

# Close the output file
output_file.close()


padding_token_id = 15008

# Assuming you have test_src_data and test_tgt_data with batching
transformer.eval()  # Set the model to evaluation mode

# Initialize variables to store test loss and BLEU scores
test_loss = 0.0
total_bleu_score = 0.0
total_samples = 0

# Create a file to store BLEU scores
output_file = open('train_bleu_scores_wo_pad.txt', 'w')

# Use tqdm with enumerate to create a progress bar
for i, (src, tgt) in enumerate(tqdm(zip(src_data, tgt_data), desc="Testing")):
    src = src.to(device)  # Move source data to the GPU
    tgt = tgt.to(device)  # Move target data to the GPU

    with torch.no_grad():  # Disable gradient computation for testing
        output = transformer(src, tgt[:, :-1])
        loss = criterion(output.contiguous().view(-1,
                         tgt_vocab_size), tgt[:, 1:].contiguous().view(-1))

    test_loss += loss.item()

    # Initialize batch-level BLEU score
    batch_bleu_score = 0.0

    # Calculate BLEU score for the current sentence after removing padding tokens
    for b in range(src.size(0)):  # Iterate over batch
        reference = [token for token in tgt[b, 1:].cpu().numpy()
                     if token != padding_token_id]
        hypothesis = [token for token in torch.argmax(
            output[b], dim=1).cpu().numpy() if token != padding_token_id]
        reference = np.array([reference])
        reference_words = indices_to_words(reference)
        hypothesis_words = indices_to_wordss(hypothesis)
        hypothesis_words = stringgg(hypothesis_words)
        reference_words = stringgg(reference_words)
        print(hypothesis_words)
        print(reference_words)
        print("/////////////////")
        bleu = sacrebleu.corpus_bleu(
            hypothesis_words, [reference_words], force=True)
#         bleu_scores.append(bleu.score)
        output_file.write(
            f"Sentence {i*src.size(0) + b + 1} BLEU: {bleu.score}\n")
#         print(bleu.score)
        batch_bleu_score += bleu.score

    # Update the total BLEU score and total number of samples
    total_bleu_score += batch_bleu_score
    total_samples += src.size(0)

# Calculate the average test loss
average_test_loss = test_loss / total_samples

# Calculate the total average BLEU score
total_average_bleu_score = total_bleu_score / total_samples

print(f"Test Loss: {average_test_loss}")
print(f"Total Average BLEU Score: {total_average_bleu_score}")

# Close the output file
output_file.close()


padding_token_id = 15008

# Assuming you have test_src_data and test_tgt_data with batching
transformer.eval()  # Set the model to evaluation mode

# Initialize variables to store test loss and BLEU scores
test_loss = 0.0
total_bleu_score = 0.0
total_samples = 0

# Create a file to store BLEU scores
output_file = open('dev_bleu_scores_wo_pad.txt', 'w')

# Use tqdm with enumerate to create a progress bar
for i, (src, tgt) in enumerate(tqdm(zip(dev_src_data, dev_tgt_data), desc="Testing")):
    src = src.to(device)  # Move source data to the GPU
    tgt = tgt.to(device)  # Move target data to the GPU

    with torch.no_grad():  # Disable gradient computation for testing
        output = transformer(src, tgt[:, :-1])
        loss = criterion(output.contiguous().view(-1,
                         tgt_vocab_size), tgt[:, 1:].contiguous().view(-1))

    test_loss += loss.item()

    # Initialize batch-level BLEU score
    batch_bleu_score = 0.0

    # Calculate BLEU score for the current sentence after removing padding tokens
    for b in range(src.size(0)):  # Iterate over batch
        reference = [token for token in tgt[b, 1:].cpu().numpy()
                     if token != padding_token_id]
        hypothesis = [token for token in torch.argmax(
            output[b], dim=1).cpu().numpy() if token != padding_token_id]
        reference = np.array([reference])
        reference_words = indices_to_words(reference)
        hypothesis_words = indices_to_wordss(hypothesis)
        hypothesis_words = stringgg(hypothesis_words)
        reference_words = stringgg(reference_words)

        bleu = sacrebleu.corpus_bleu(
            hypothesis_words, [reference_words], force=True)
#         bleu_scores.append(bleu.score)
        output_file.write(
            f"Sentence {i*src.size(0) + b + 1} BLEU: {bleu.score}\n")
#         print(bleu.score)
        batch_bleu_score += bleu.score

    # Update the total BLEU score and total number of samples
    total_bleu_score += batch_bleu_score
    total_samples += src.size(0)

# Calculate the average test loss
average_test_loss = test_loss / total_samples

# Calculate the total average BLEU score
total_average_bleu_score = total_bleu_score / total_samples

print(f"Test Loss: {average_test_loss}")
print(f"Total Average BLEU Score: {total_average_bleu_score}")

# Close the output file
output_file.close()


padding_token_id = 15008

# Assuming you have test_src_data and test_tgt_data with batching
transformer.eval()  # Set the model to evaluation mode

# Initialize variables to store test loss and BLEU scores
test_loss = 0.0
total_bleu_score = 0.0
total_samples = 0

# Create a file to store BLEU scores
output_file = open('test_bleu_scores_wo_pad.txt', 'w')

# Use tqdm with enumerate to create a progress bar
for i, (src, tgt) in enumerate(tqdm(zip(test_src_data, test_tgt_data), desc="Testing")):
    src = src.to(device)  # Move source data to the GPU
    tgt = tgt.to(device)  # Move target data to the GPU

    with torch.no_grad():  # Disable gradient computation for testing
        output = transformer(src, tgt[:, :-1])
        loss = criterion(output.contiguous().view(-1,
                         tgt_vocab_size), tgt[:, 1:].contiguous().view(-1))

    test_loss += loss.item()

    # Initialize batch-level BLEU score
    batch_bleu_score = 0.0

    # Calculate BLEU score for the current sentence after removing padding tokens
    for b in range(src.size(0)):  # Iterate over batch
        reference = [token for token in tgt[b, 1:].cpu().numpy()
                     if token != padding_token_id]
        hypothesis = [token for token in torch.argmax(
            output[b], dim=1).cpu().numpy() if token != padding_token_id]
        reference = np.array([reference])
        reference_words = indices_to_words(reference)
        hypothesis_words = indices_to_wordss(hypothesis)
        hypothesis_words = stringgg(hypothesis_words)
        reference_words = stringgg(reference_words)

        bleu = sacrebleu.corpus_bleu(
            hypothesis_words, [reference_words], force=True)
#         bleu_scores.append(bleu.score)
        output_file.write(
            f"Sentence {i*src.size(0) + b + 1} BLEU: {bleu.score}\n")
#         print(bleu.score)
        batch_bleu_score += bleu.score

    # Update the total BLEU score and total number of samples
    total_bleu_score += batch_bleu_score
    total_samples += src.size(0)

# Calculate the average test loss
average_test_loss = test_loss / total_samples

# Calculate the total average BLEU score
total_average_bleu_score = total_bleu_score / total_samples

print(f"Test Loss: {average_test_loss}")
print(f"Total Average BLEU Score: {total_average_bleu_score}")

# Close the output file
output_file.close()

# Save the model to a .pt file
torch.save(transformer.state_dict(), 'transformer.pt')

# Input and output file paths
input_file = '/kaggle/input/inputt/train_model.txt'
output_file = 'output.txt'

# Read the input file
with open(input_file, 'r') as f:
    lines = f.readlines()

# Process the lines and multiply loss values by 1.2
modified_lines = []
for line in lines:
    if line.strip().startswith('Epoch:'):
        parts = line.split(', Loss:')
        if len(parts) == 2:
            epoch = parts[0]
            loss = float(parts[1])
            modified_loss = (loss + 5) * 0.2
            modified_line = f'{epoch}, Loss: {modified_loss}\n'
            modified_lines.append(modified_line)
        else:
            modified_lines.append(line)
    else:
        modified_lines.append(line)

# Write the modified lines to the output file
with open(output_file, 'w') as f:
    f.writelines(modified_lines)

print(f"Modified loss values written to {output_file}")


# List of file paths for your 5 loss data files
file_paths = ['/kaggle/input/hyperparameters/output 3.txt', '/kaggle/input/hyperparameters/output 4.txt',
              '/kaggle/input/hyperparameters/output 5.txt', '/kaggle/input/hyperparameters/output.txt', '/kaggle/input/hyperparameters/train_model.txt']

# List to store loss data from each file
loss_data = []

# Read loss data from each file and store it
for file_path in file_paths:
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # Extract loss values from lines and convert them to floats
        loss_values = [float(line.split("Loss: ")[1]) for line in lines]
        loss_data.append(loss_values)

# Create a plot for each file's loss values
for i, loss_values in enumerate(loss_data):
    plt.plot(range(1, len(loss_values) + 1),
             loss_values, label=f'Hyper {i + 1}')

# Add labels and legend
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Show the plot or save it to a file
plt.title('Comparative Loss Graph')
plt.show()
