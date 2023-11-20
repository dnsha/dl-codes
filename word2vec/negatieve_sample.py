import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset


class SkipGramDataset(Dataset):

    def __init__(self, context_size, corpus_size, corpus, word_dict):
        self.X = []
        self.y = []
        for i in range(context_size, corpus_size - context_size):
            for j in range(1, context_size + 1):
                self.X.append(word_dict[corpus[i]])
                self.y.append(word_dict[corpus[i - j]])
                self.X.append(word_dict[corpus[i]])
                self.y.append(word_dict[corpus[i + j]])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return torch.tensor(self.X[index]), torch.tensor(self.y[index])


class SkipGramNg(nn.Module):
    def __init__(self, vocab_size, embed_size, noise_dist=None):
        super(SkipGramNg, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.input_embed = nn.Embedding(vocab_size, embed_size)
        self.output_embed = nn.Embedding(vocab_size, embed_size)
        self.noise_dist = noise_dist

    def forward_input(self, input_wprds):
        return self.input_embed(input_wprds)

    def forward_output(self, output_words):
        return self.output_embed(output_words)

    def forward_noise(self, batch_size, n_samples):
        if self.noise_dist is None:
            noise_dist = torch.ones(self.vocab_size)
        else:
            noise_dist = self.noise_dist

        return self.output_embed(torch.multinomial(noise_dist,
                                                   batch_size * n_samples,
                                                   replacement=True)).view(batch_size, n_samples, self.embed_size)


class NegSampLoss(nn.Module):
    def __init__(self):
        super(NegSampLoss, self).__init__()

    def forward(self, input_embed, output_embed, noise):
        batch_size, embed_size = input_embed.shape
        input_embed = input_embed.view(batch_size, embed_size, 1)

        output_embed = output_embed.view(batch_size, 1, embed_size)

        out_loss = torch.bmm(output_embed, input_embed).sigmoid().log()

        out_loss = out_loss.squeeze()

        noise_loss = torch.bmm(noise.neg(), input_embed).sigmoid().log()
        noise_loss = noise_loss.squeeze().sum(1)

        return -(out_loss + noise_loss).mean()


if __name__ == "__main__":
    sentences = [
        "The sun rises in the east",
        "Dogs love to chase after tennis balls",
        "Raindrops fell softly on the windowpane",
        "She always carries a book in her bag",
        "The smell of freshly baked bread filled the air",
        "I can not believe it is already December",
        "The mountains are covered in snow",
        "He played the guitar by the campfire",
        "The baby giggled as the puppy licked her face",
        "The city skyline was illuminated with colorful lights",
        "The old oak tree stood tall in the park",
        "I need to buy some groceries after work",
        "The ocean waves crashed against the shore",
        "Birds chirped in the early morning",
        "My favorite color is blue",
        "The movie kept us on the edge of our seats",
        "She wore a beautiful dress to the party",
        "The car sped down the highway",
        "A cup of hot cocoa warms you up on a cold day",
        "I am planning a trip to Europe next summer",
        "Flowers bloomed in the garden",
        "He always tells the funniest jokes",
        "The cat stretched lazily in the sun",
        "The restaurant serves delicious sushi",
        "We hiked to the top of the mountain",
        "The children played in the park all day",
        "The stars twinkled in the night sky",
        "She received a bouquet of roses for her birthday",
        "Reading a good book is my favorite pastime",
        "The alarm clock woke me up early",
        "The basketball game went into overtime",
        "I can not find my keys anywhere",
        "The butterfly fluttered from flower to flower",
        "The museum had a fascinating exhibit on ancient Egypt",
        "The teacher explained the lesson to the students",
        "The river flowed gently through the forest",
        "The chef prepared a delicious three-course meal",
        "He took a deep breath and jumped into the pool",
        "The snowflakes melted on my tongue",
        "The fireworks lit up the night sky",
        "We went for a walk along the beach",
        "The puzzle was missing a few pieces",
        "She wrote a heartfelt letter to her best friend",
        "The clock on the wall ticked loudly",
        "The scientist conducted experiments in the lab",
        "I love the sound of the rain on the roof",
        "The marathon runners crossed the finish line",
        "The concert was sold out",
        "We watched a documentary about wildlife conservation",
        "The concert was sold out"
    ]

    corpus = " ".join(sentences).split()
    vocab = list(set(corpus))
    vocab_size = len(vocab)
    word_dict = {w: i for i, w in enumerate(vocab)}
    corpus_size = len(corpus)
    context_size = 2
    batch_size = 5
    skipGramDataset = SkipGramDataset(context_size, corpus_size, corpus, word_dict)
    dataloader = torch.utils.data.DataLoader(skipGramDataset, batch_size=batch_size, shuffle=True)
    model = SkipGramNg(vocab_size,embed_size=100)
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    criterion = NegSampLoss()
    for epoch in range(10):
        for word, label in dataloader:
            optimizer.zero_grad()
            input = model.input_embed(word)
            output = model.output_embed(word)
            noise = model.forward_noise(input.shape[0],3)
            loss = criterion(input, output, noise)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 2 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

    print(model.forward_input(torch.tensor(word_dict[vocab[5]])))





