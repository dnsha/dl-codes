import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import T_co


class SkipGramDataset(Dataset):

    def __init__(self, vocab_size, training_data):
        self.one_hot = torch.eye(vocab_size)
        self.X = []
        self.y = []
        for tup in training_data:
            self.X.append(torch.tensor(self.one_hot[tup[0]]))
            self.y.append(torch.tensor(tup[1]))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class SkipGramModel(nn.Module):

    def __init__(self, vocab_size, embed_size=100):
        super(SkipGramModel, self).__init__()
        self.w = nn.Linear(vocab_size, embed_size, bias=False)
        self.wt = nn.Linear(embed_size, vocab_size, bias=False)

    def forward(self, input):
        return self.wt(self.w(input))


if __name__ == "__main__":
    context_size = 2
    sents = ["Hi All",
             "How are you",
             "Today is a great day",
             "you are awesome"]

    corpus = " ".join(sents).split()
    vocab = list(set(corpus))
    word_dict = {w:i for i,w in enumerate(vocab)}
    training_data = []
    for i in range(context_size,len(corpus)-context_size):
        for j in range(1,context_size+1):
            training_data.append([word_dict[corpus[i]],word_dict[corpus[i-j]]])
            training_data.append([word_dict[corpus[i]], word_dict[corpus[i + j]]])

    dataset = SkipGramDataset(len(vocab), training_data)
    training_generator = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    skip_gram_model = SkipGramModel(len(vocab))

    optimizer = optim.Adam(skip_gram_model.parameters(),lr = 0.001)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(50):
        for train_batch, train_lebel in training_generator:
            optimizer.zero_grad()
            output = skip_gram_model(train_batch)
            loss = criterion(output,train_lebel )

            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
    params = skip_gram_model.parameters()
    params = [p.clone().detach() for p in params]
    word_vec = {}
    for index,word in enumerate(vocab):
        word_vec[word] = params[0][:,index]
    print(word_vec)