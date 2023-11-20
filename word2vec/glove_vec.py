import torch
import torch.nn as nn
import torch.optim as optim


class Glove(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(Glove, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.tilda_embed = nn.Embedding(vocab_size, embed_size)
        self.bias = nn.Embedding(vocab_size, 1)
        self.tilda_bias = nn.Embedding(vocab_size, 1)

    def forward(self, input_word, target_word):

        return torch.sum(self.embed(input_word) * self.tilda_embed(target_word), dim=1) + self.bias(
            input_word) + self.tilda_bias(target_word)

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
    context_size = 2
    corpus = " ".join(sentences).split()
    vocab = list(set(corpus))
    word_dict = {w:i for i,w in enumerate(vocab)}
    vocab_size = len(vocab)
    training_data = [[0] * vocab_size] * vocab_size
    for i in range(context_size, len(corpus) - context_size):
        for j in range(1, context_size + 1):

            training_data[word_dict[corpus[i]]][word_dict[corpus[i - j]]] += 1
            training_data[word_dict[corpus[i]]][word_dict[corpus[i + j]]] += 1

    model = Glove(vocab_size,5)
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    criterian = nn.MSELoss()
    for epoch in range(10):
        total_loss = 0
        for i in range(vocab_size):
            for j in range(vocab_size):
                if training_data[i][j] > 0:
                    optimizer.zero_grad()
                    output = model(torch.tensor([i]), torch.tensor([j]))
                    loss = criterian(output,torch.log(torch.tensor([training_data[i][j]])))
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
        print("loss after {} is {}".format(epoch,total_loss))

    for word in vocab:
        print(word,model.embed(torch.tensor(word_dict[word]).detach()))
