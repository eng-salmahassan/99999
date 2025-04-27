#!/usr/bin/env python
# coding: utf-8

# In[2]:


import random
import math
words = ["I", "love", "Computational", "Neuroscience"]
vocab = list(set(words))
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for word, i in word_to_idx.items()}

input_seq = [word_to_idx["I"], word_to_idx["love"], word_to_idx["Computational"]]
target = word_to_idx["Neuroscience"]

hidden_size = 5
vocab_size = len(vocab)

Wxh = [[(random.random() - 0.5) for _ in range(hidden_size)] for _ in range(vocab_size)]
Whh = [[(random.random() - 0.5) for _ in range(hidden_size)] for _ in range(hidden_size)]
Why = [[(random.random() - 0.5) for _ in range(vocab_size)] for _ in range(hidden_size)] 
bh = [(random.random() - 0.5) for _ in range(hidden_size)]
by = [(random.random() - 0.5) for _ in range(vocab_size)]  


def tanh(x):
    return math.tanh(x)

def dtanh(x):
    return 1 - math.tanh(x) ** 2

def softmax(x):
    exps = [math.exp(i) for i in x]
    sum_exps = sum(exps)
    return [j / sum_exps for j in exps]

def forward(inputs):
    hs = [{i: 0 for i in range(hidden_size)}]
    for t in range(len(inputs)):
        x = [0] * vocab_size
        x[inputs[t]] = 1
        
        h_new = {}
        for i in range(hidden_size):
            total = bh[i]
            for j in range(vocab_size):
                total += x[j] * Wxh[j][i]
            for j in range(hidden_size):
                total += hs[-1][j] * Whh[j][i]
            h_new[i] = tanh(total)
        hs.append(h_new)

    
    y = [by[i] for i in range(vocab_size)]
    for k in range(vocab_size):
        for i in range(hidden_size):
            y[k] += hs[-1][i] * Why[i][k]
    return y, hs

def backward(inputs, target, learning_rate=0.01):
    y_pred, hs = forward(inputs)
    
    preds = softmax(y_pred)
    loss = -math.log(preds[target])
    print("Loss:", loss)

    dy = preds.copy()
    dy[target] -= 1

    dWxh = [[0 for _ in range(hidden_size)] for _ in range(vocab_size)]
    dWhh = [[0 for _ in range(hidden_size)] for _ in range(hidden_size)]
    dWhy = [[0 for _ in range(vocab_size)] for _ in range(hidden_size)]
    dbh = [0 for _ in range(hidden_size)]
    dby = [0 for _ in range(vocab_size)]

    for i in range(hidden_size):
        for j in range(vocab_size):
            dWhy[i][j] += hs[-1][i] * dy[j]
    for j in range(vocab_size):
        dby[j] += dy[j]

    dh = {i: 0 for i in range(hidden_size)}
    for i in range(hidden_size):
        for j in range(vocab_size):
            dh[i] += Why[i][j] * dy[j]

    for t in reversed(range(len(inputs))):
        dhraw = {i: dh[i] * dtanh(hs[t+1][i]) for i in range(hidden_size)}
        x = [0] * vocab_size
        x[inputs[t]] = 1
        for i in range(hidden_size):
            dbh[i] += dhraw[i]
            for j in range(vocab_size):
                dWxh[j][i] += x[j] * dhraw[i]
            for j in range(hidden_size):
                dWhh[j][i] += hs[t][j] * dhraw[i]
        
        new_dh = {i: 0 for i in range(hidden_size)}
        for i in range(hidden_size):
            for j in range(hidden_size):
                new_dh[i] += Whh[i][j] * dhraw[j]
        dh = new_dh

   
    for i in range(vocab_size):
        for j in range(hidden_size):
            Wxh[i][j] -= learning_rate * dWxh[i][j]
    for i in range(hidden_size):
        for j in range(hidden_size):
            Whh[i][j] -= learning_rate * dWhh[i][j]
    for i in range(hidden_size):
        for j in range(vocab_size):
            Why[i][j] -= learning_rate * dWhy[i][j]
    for i in range(hidden_size):
        bh[i] -= learning_rate * dbh[i]
    for i in range(vocab_size):
        by[i] -= learning_rate * dby[i]

for epoch in range(100):
    backward(input_seq, target)

def predict(inputs):
    y_pred, _ = forward(inputs)
    preds = softmax(y_pred)
    pred_idx = preds.index(max(preds))
    return idx_to_word[pred_idx]

predicted_word = predict(input_seq)
print("\nPredicted word:", predicted_word)
print("Actual word:", idx_to_word[target])


# In[ ]:




