import random

import torch
import torch.nn as nn
import torch.optim as optim

class Doc2VecDM(nn.Module):
    def __init__(self, vocab_size, doc_size, embed_size):
        super(Doc2VecDM, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        self.doc_embeddings = nn.Embedding(doc_size, embed_size)
        self.linear = nn.Linear(embed_size, vocab_size)

    def forward(self, input_words, input_docs):
        target_embed = self.word_embeddings(input_words[0])
        for w in input_words[1:]:
            target_embed += self.word_embeddings(w)
        doc_embeddings = self.doc_embeddings(input_docs).squeeze(0)
        return self.linear(torch.add(target_embed, doc_embeddings)).unsqueeze(0)




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
    context_size = 2
    model = Doc2VecDM(vocab_size,  len(sents), embed_size)
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    criterian = nn.CrossEntropyLoss()
    print(vocab_size)
    for epoch in range(10):
        total_loss = 0
        for doc_id, doc_words in enumerate(sents_coded):
            for i in range(len(doc_words)-context_size):
                optimizer.zero_grad()
                input_words = []
                label = torch.tensor([doc_words[i]], dtype=torch.long)
                for j in range(1,context_size+1):
                    input_words.append(doc_words[i-j])
                    input_words.append(doc_words[i+j])
                input_words = torch.tensor(input_words, dtype=torch.long)

                input_doc = torch.tensor([doc_id], dtype=torch.long)
                output = model(input_words, input_doc)
                loss = criterian(output,label)
                total_loss += loss
                loss.backward()
                optimizer.step()
        print("loss after {} epoch = {}".format(epoch, total_loss))
    for i,sent in enumerate(sents):
        print(sent,model.doc_embeddings(torch.tensor([i])).detach())



