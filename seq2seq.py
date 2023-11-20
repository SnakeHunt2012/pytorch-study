#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import time
import math
import random
import unicodedata
import numpy as np

import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

# Official Impl
import torch.nn.functional as F

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device {device}")

#################
### Load Data ###
#################

MAX_LENGTH = 10
eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

BOS_TOKEN = 0
EOS_TOKEN = 1

class Vocabulary:
    
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "BOS", 1: "EOS"}
        self.vocabulary_size = 2 # BOS and EOS

    def add_word(self, word):
        if word in self.word2index:
            self.word2count[word] += 1
        else:
            self.word2index[word] = self.vocabulary_size
            self.word2count[word] = 1
            self.index2word[self.vocabulary_size] = word
            self.vocabulary_size += 1

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)


def unicode2ascii(string):
    
    return "".join(char for char in unicodedata.normalize("NFD", string) if unicodedata.category(char) != "Mn")


def normalize_string(string):
    
    string = unicode2ascii(string.lower().strip())
    string = re.sub(r"([.!?])", r" \1", string)
    string = re.sub(r"[^a-zA-Z!?]+", r" ", string)
    return string.strip()


def read_vocabulary(lang1, lang2, reverse = False):

    print("Reading lines ...")
    line_list = open("data/%s-%s.txt" % (lang1, lang2), encoding = 'utf-8').read().strip().split('\n')
    pair_list = [[normalize_string(sentence) for sentence in line.split('\t')] for line in line_list]
    
    if reverse:
        pair_list = [[pair[1], pair[0]] for pair in pair_list]
        source_vocabulary = Vocabulary(lang2)
        target_vocabulary = Vocabulary(lang1)
    else:
        source_vocabulary = Vocabulary(lang1)
        target_vocabulary = Vocabulary(lang2)

    return source_vocabulary, target_vocabulary, pair_list


def filter_pair(pair):
    
    return len(pair[0].split(' ')) < MAX_LENGTH and len(pair[1].split(' ')) < MAX_LENGTH and pair[1].startswith(eng_prefixes)


def filter_pair_list(pair_list):
    
    return [pair for pair in pair_list if filter_pair(pair)]


def preprocess_data(lang1, lang2, reverse = False):

    source_vocabulary, target_vocabulary, pair_list = read_vocabulary(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pair_list))
    
    pair_list = filter_pair_list(pair_list)
    print("Trimmed to %s sentences pairs:" % len(pair_list))
    
    print("Build vocabularies ...")
    for pair in pair_list:
        source_vocabulary.add_sentence(pair[0])
        target_vocabulary.add_sentence(pair[1])
        
    print("Built vocabularies:")
    print(source_vocabulary.name, source_vocabulary.vocabulary_size)
    print(target_vocabulary.name, target_vocabulary.vocabulary_size)

    return source_vocabulary, target_vocabulary, pair_list
    

source_vocabulary, target_vocabulary, pair_list = preprocess_data("eng", "fra", True)
print(random.choice(pair_list))

def indices_from_sentence(vocabulary, sentence):

    return [vocabulary.word2index[word] for word in sentence.split(' ')] # TODO: handle un-known words


def tensor_from_sentence(vocabulary, sentence):

    indices = indices_from_sentence(vocabulary, sentence)
    indices.append(EOS_TOKEN)
    return torch.tensor(indices, dtype = torch.long, device = device).view(1, -1)


# (Not Used)
# def tensor_from_pair(vocabulary, pair):
# 
#     source_tensor = tensor_from_sentence(vocabulary, pair[0])
#     target_tensor = tensor_from_sentence(vocabulary, pair[1])
#     return source_tensor, target_tensor


def get_dataloader(batch_size):

    source_vocabulary, target_vocabulary, pair_list = preprocess_data("eng", "fra", True)

    pair_length = len(pair_list)
    source_indices_np = np.zeros((pair_length, MAX_LENGTH), dtype = np.int32)
    target_indices_np = np.zeros((pair_length, MAX_LENGTH), dtype = np.int32)

    for index, (source_sentence, target_sentence) in enumerate(pair_list):
        source_indices = indices_from_sentence(source_vocabulary, source_sentence)
        target_indices = indices_from_sentence(target_vocabulary, target_sentence)
        source_indices.append(EOS_TOKEN)
        target_indices.append(EOS_TOKEN)
        source_indices_np[index, :len(source_indices)] = source_indices
        target_indices_np[index, :len(target_indices)] = target_indices

    dataset = TensorDataset(
        torch.LongTensor(source_indices_np).to(device),
        torch.LongTensor(target_indices_np).to(device)
    )

    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler = sampler, batch_size = batch_size)
    return source_vocabulary, target_vocabulary, dataloader

###################
### RNN Encoder ###
###################

class EncoderRNN(nn.Module):

    def __init__(self, vocabulary_size, hidden_size, dropout = 0.1):

        super(EncoderRNN, self).__init__()
        
        self.embedding = nn.Embedding(vocabulary_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first = True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        
        output = self.dropout(self.embedding(input))
        output, hidden = self.gru(output)
        return output, hidden

###################
### RNN Decoder ###
###################

class DecoderRNN(nn.Module):

    def __init__(self, vocabulary_size, hidden_size):

        super(DecoderRNN, self).__init__()

        self.embedding = nn.Embedding(vocabulary_size, hidden_size)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first = True)
        self.linear = nn.Linear(hidden_size, vocabulary_size)
        self.softmax = nn.LogSoftmax(dim = -1)

    def forward_step(self, input, hidden):
        
        output = self.embedding(input)
        #output = self.relu(output)
        output = F.relu(output) # F.relu() is better than nn.relu(), I just don't know why.
        output, hidden = self.gru(output, hidden)
        output = self.linear(output)
        return output, hidden

    def forward(self, input, hidden, target = None): # input -> decoder_outputs, hidden -> decoder_hidden

        batch_size = input.size(0)
        input = torch.empty(batch_size, 1, dtype = torch.long, device = device).fill_(BOS_TOKEN)
        output = []
        
        for i in range(MAX_LENGTH):
            output_step, hidden = self.forward_step(input, hidden)
            output.append(output_step)

            if target is not None:
                # Teacher forcing: Feed the target as the next input
                input = target[:, i].unsqueeze(1)
            else:
                # Without teacher forcing: use its own predictions as the next input
                input = output_step.topk(1)[1].squeeze(-1).detach() # topk -> (values, indices)

        output = torch.cat(output, dim = 1)
        output = self.softmax(output)
        return output, hidden, None


#######################
### Train RNN Model ###
#######################

def as_minutes(seconds):
    minutes = math.floor(seconds / 60)
    seconds -= minutes * 60
    return '%dm %ds' % (minutes, seconds)


def time_since(since, percent):

    now = time.time()
    seconds = now - since
    es = seconds / (percent)
    rs = es - seconds
    return "%s (- %s)" % (as_minutes(seconds), as_minutes(rs))


def train_epoch(dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, debug = False):

    total_loss = 0
    for data in dataloader:
        input_tensor, target_tensor = data

        encoder_output, encoder_hidden = encoder(input_tensor)
        # teacher forcing
        decoder_output, _, _ = decoder(encoder_output, encoder_hidden, target_tensor)
        # without teacher forcing
        #decoder_output, _, _ = decoder(encoder_output, encoder_hidden)

        if debug:
            print(f"input_tensor: {input_tensor}, size: {input_tensor.size()}")
            print(f"encoder_hidden: {encoder_hidden}, size: {encoder_hidden.size()}")
            print(f"decoder_output.view(-1, decoder_output.size(-1)): {decoder_output.view(-1, decoder_output.size(-1))}, size: {decoder_output.view(-1, decoder_output.size(-1)).size()}")
            print(f"target_tensor.view(-1): {target_tensor.view(-1)}, size: {target_tensor.view(-1).size()}")
            

        loss = criterion(
            decoder_output.view(-1, decoder_output.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train(dataloader, encoder, decoder, n_epochs, learning_rate = 0.0001, print_every = 100, plot_every = 100):

    start = time.time()
    plot_losses = []
    print_loss_total = 0 # Reset every print every
    plot_loss_total = 0 # Reset every plot every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr = learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr = learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)    
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print("%s (%d %d%%) %.4f" % (
                time_since(start, epoch / n_epochs),
                epoch,
                epoch / n_epochs * 100,
                print_loss_avg
            ))
            print_loss_total = 0

        #if epoch % plot_every == 0:
        #    plot_loss_avg = plot_loss_total / plot_every
        #    plot_losses.append(plot_loss_avg)
        #    plot_loss_total = 0

def evaluate(sentence, encoder, decoder, source_vocabulary, target_vocabulary):
    with torch.no_grad():
        tensor = tensor_from_sentence(source_vocabulary, sentence)
        encoder_output, encoder_hidden = encoder(tensor)
        decoder_output, decoder_hidden, _ = decoder(encoder_output, encoder_hidden)
        
        _, topi = decoder_output.topk(1)
        decoder_ids = topi.squeeze()

        word_list = []
        for index in decoder_ids:
            if index.item == EOS_TOKEN:
                word_list.append('<EOS>')
                break
            word_list.append(target_vocabulary.index2word[index.item()])
    return " ".join(word_list)


hidden_size = 128
batch_size = 32

source_vocabulary, target_vocabulary, train_dataloader = get_dataloader(batch_size)
encoder = EncoderRNN(source_vocabulary.vocabulary_size, hidden_size).to(device)
decoder = DecoderRNN(target_vocabulary.vocabulary_size, hidden_size).to(device)

train(train_dataloader, encoder, decoder, 80, print_every = 1, plot_every = 1)

encoder.eval()
decoder.eval()
print(evaluate("il n est pas aussi grand que son pere", encoder, decoder, source_vocabulary, target_vocabulary))
print(evaluate("je suis trop fatigue pour conduire", encoder, decoder, source_vocabulary, target_vocabulary))
print(evaluate("je suis desole si c est une question idiote", encoder, decoder, source_vocabulary, target_vocabulary))
