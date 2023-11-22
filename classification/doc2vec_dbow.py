import torch
import random
import torch.nn as nn
import torch.optim as optim


class Doc2VecDBOW(nn.Module):
    def __init__(self, vocab_size, embed_size, doc_size):
        super(Doc2VecDBOW, self).__init__()
        self.doc_embeddings = nn.Embedding(doc_size, embed_size)
        self.linear = nn.Linear(embed_size, vocab_size)

    def forward(self, docs):
        return self.linear(self.doc_embeddings(docs))


if __name__ == "__main__":
    sents = [
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
        "We watched a documentary about wildlife conservation"
    ]

    corpus = " ".join(sents).split()
    vocab = list(set(corpus))
    vocab = {w: i for i, w in enumerate(vocab)}

    sents_coded = [[vocab[word] for word in sent.split()] for sent in sents]

    vocab_size = len(vocab)
    embed_size = 3
    model = Doc2VecDBOW(vocab_size, embed_size, len(sents))
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    criterian = nn.CrossEntropyLoss()
    for epoch in range(10):
        total_loss = 0
        for doc_id, doc_words in enumerate(sents_coded):
            if doc_words:
                word_id = random.choice(doc_words)
                word_tensor = torch.tensor([word_id], dtype=torch.long)
                doc_tensor = torch.tensor([doc_id], dtype=torch.long)
                # negative_samples = negative_sampling([word_id], 1, vocab_szie)
                optimizer.zero_grad()
                output = model(doc_tensor)
                loss = criterian(output, word_tensor)
                loss.backward()
                optimizer.step()
                total_loss += loss
        print("loss after {} epoch = {}".format(epoch, total_loss))

    for i, sent in enumerate(sents):
        print(sent, model.doc_embeddings(torch.tensor([i])).detach())
