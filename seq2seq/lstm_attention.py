import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class TranslationDataset(Dataset):
    def __init__(self, src_data, tgt_data):
        self.src_data = src_data
        self.tgt_data = tgt_data

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        src = torch.tensor(self.src_data[idx])
        tgt = torch.tensor(self.tgt_data[idx])
        return src, tgt




class EncoderLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, dropout=0.1):
        super(EncoderLSTM, self).__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.encoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_input):
        # encoder input  (batch_size, seq_len)
        encoder_input = self.dropout(self.embed(encoder_input))  # encoder input (batch_size, seq_len, hidden_size)
        encoder_output, (encoder_hidden, encoder_cell) = self.encoder(encoder_input)
        # encoder_output (batch_size, seq_len, hidden_size)
        # encoder_hidden (D * num_layers, batch_size, hidden_size)
        return encoder_output, encoder_hidden


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, key):
        # query comes from decoder hidden state (batch_size, num_layer * D, hidden_dim)
        # key comes from encoder output (batch_size, seq_len, hidden_dim)
        scores = self.Va(torch.tanh(self.Ua(query) + self.Ua(key)))
        # scores : (batch_size, seq_len, 1)
        scores = scores.permute(0, 2, 1)  # change scores dimention to apply softmax on seq len (batch_size, 1, seq_len)
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, key)  # context dim (batch_size, 1, hidden_size)
        return context, weights


class DecoderLSTM(nn.Module):
    def __init__(self, hidden_dim, vocab_size, dropout=0.1):
        super(DecoderLSTM, self).__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.decoder = nn.LSTM(2 * hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim,vocab_size)
        self.attn = Attention(hidden_dim)

    def forward(self, encoder_output, encoder_hidden, target_tensor=None):
        batch_size = encoder_output.shape[0]
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long).fill_(0) # decoder input (batch_size,1)
        decoder_hidden = encoder_hidden
        seq_len = encoder_output.shape[1]
        decoder_outputs = []
        attn_weights = []

        for i in range(seq_len):
            query = decoder_hidden.permute(1, 0, 2)  # query (batch_size, D* num_layers, hidden_size)

            context, attn_weight = self.attn(query, encoder_output)
            decoder_input = self.dropout(self.embed(decoder_input))
            decoder_output, (decoder_hidden, decoder_Cell) = self.decoder(torch.cat((context, decoder_input), dim=2))
            decoder_output = self.out(decoder_output)
            decoder_outputs.append(decoder_output)
            attn_weights.append(attn_weight)
            if target_tensor is not None:
                decoder_input = target_tensor[:,i].unsqueeze(1)
            else:
                _, topi = decoder_output.topk(1)
                decoder_input= topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs,dim=1) # decoder outputs (batch_size, seq_len, hidden_dim)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attn_weights = torch.cat(attn_weights, dim=1) # attention outputs (batch_size, seq_len, hidden_dim)
        return decoder_outputs, decoder_hidden, attn_weights


sentence_pairs = [
    ("Hello how are you", "Bonjour comment ça va"),
    ("I am learning French", "J apprends le français"),
    ("What is your name", "Comment tu t appelles"),
    ("I love reading books", "J aime lire des livres."),
    ("The weather is nice today.", "Il fait beau aujourd hui")
]

source_vocab = set()
for sent in sentence_pairs:
    for word in sent[0].split():
        source_vocab.add(word)
source_vocab.add("<pad>")
source_vocab = {w:i for i,w in enumerate(list(source_vocab))}


target_vocab = set()
for sent in sentence_pairs:
    for word in sent[1].split():
        target_vocab.add(word)
target_vocab.add("<pad>")
target_vocab = {w:i for i,w in enumerate(list(target_vocab))}


src_data = [[source_vocab[w] for w in s.split()] for (s,_) in sentence_pairs]
trg_data = [[target_vocab[w] for w in t.split()] for (_, t) in sentence_pairs]
src_Seq_len = 7
target_seq_len = 7

for i,data in enumerate(src_data):
    if len(data) < src_Seq_len:
        src_data[i] = src_data[i] + [source_vocab["<pad>"]]  * (src_Seq_len - len(data))
    else:
        src_data[i] = data[:src_Seq_len]


for i,data in enumerate(trg_data):
    if len(data) < target_seq_len:
        trg_data[i] = trg_data[i] + [target_vocab["<pad>"]]  * (target_seq_len - len(data))
    else:
        trg_data[i] = data[:target_seq_len]



TranslationDataset(src_data, trg_data)
dataset = TranslationDataset(src_data, trg_data)
loader = DataLoader(dataset, batch_size=2, shuffle=True)
input_dim = len(source_vocab)
output_dim = len(target_vocab)

enc_embed_dim = 32
dec_embed_dim = 32
hid_Dim = 16
num_layers = 2
enc_dropout = 0.1
dec_dropout = 0.1

encoder = EncoderLSTM(input_dim,hid_Dim)

decoder =  DecoderLSTM(hid_Dim, output_dim)



# Loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=target_vocab['<pad>'])
encoder_optimizer = torch.optim.Adam(encoder.parameters())

decoder_optimizer = torch.optim.Adam(decoder.parameters())

# Training loop
num_epochs = 1
for epoch in range(num_epochs):
    epoch_loss = 0

    for src, tgt in loader:
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        encoder_outputs, encoder_hidden = encoder(src)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, tgt)
        # output = model(src, tgt[:-1])


        # Flatten output for loss calculation
        output_dim = decoder_outputs.shape[-1]
        output = decoder_outputs.view(-1, output_dim)
        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            tgt.view(-1)
        )
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(loader)}")





