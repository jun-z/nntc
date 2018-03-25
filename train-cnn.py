import re
import math
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from torchtext import data

from models import CNNClassifier
from utils import pad_shorties, calc_loss, predict


# Command-line arguments.
parser = argparse.ArgumentParser(description='Train a model.')

parser.add_argument(
    '--train_file', required=True, help='training file path')

parser.add_argument(
    '--test_file', required=True, help='testing file path')

parser.add_argument(
    '--token_regex', default='\w+', help='tokenizing regex')

parser.add_argument(
    '--num_epochs', default=10, type=int, help='number of epochs')

parser.add_argument(
    '--batch_size', default=128, type=int, help='batch size')

parser.add_argument(
    '--learning_rate', default=1e-3, type=float, help='learning rate')

parser.add_argument(
    '--print_every', default=100, type=int, help='print every n iterations')

parser.add_argument(
    '--embedding_dim', default=300, type=int, help='embedding dimension')

parser.add_argument(
    '--pretrained_embeddings', type=str, help='pretrained embeddings')

parser.add_argument(
    '--filter_mapping', default='{1: 128, 2: 128}', help='mapping for filters')

parser.add_argument(
    '--dropout_prob', default=.5, type=float, help='dropout probability')

parser.add_argument(
    '--disable_cuda', action='store_true', help='disable cuda')

parser.add_argument(
    '--device_id', default=0, type=int, help='id for cuda device')

parser.add_argument(
    '--data_parallel', action='store_true', help='use data parallel')

parser.add_argument(
    '--device_ids', default='[0, 1]', help='device ids for data parallel')

args = parser.parse_args()
args.cuda = not args.disable_cuda and torch.cuda.is_available()


# Prepare training and testing data.
MIN_LEN = max(eval(args.filter_mapping).keys())

WORD = re.compile(args.token_regex)

TEXT = data.Field(lower=True,
                  tokenize=WORD.findall,
                  batch_first=True,
                  preprocessing=lambda x: pad_shorties(x, MIN_LEN))

LABEL = data.Field(sequential=False, unk_token=None)

fields = [
    ('label', LABEL),
    ('text', TEXT)
]

train_set = data.TabularDataset(args.train_file, 'csv', fields=fields)
test_set = data.TabularDataset(args.test_file, 'csv', fields=fields)

print(f'Loaded training data: {args.train_file}')
print(f'Loaded testing data: {args.test_file}')

TEXT.build_vocab(train_set,
                 min_freq=5,
                 vectors=args.pretrained_embeddings)

LABEL.build_vocab(train_set)

print(f'Number of training examples: {len(train_set.examples)}')
print(f'Number of testing examples: {len(test_set.examples)}')

print(f'Size of vocabulary: {len(TEXT.vocab)}')
print(f'Number of labels: {len(LABEL.vocab)}')

# Initiate criterion, classifier, and optimizer.
classifier = CNNClassifier(vocab_size=len(TEXT.vocab),
                           label_size=len(LABEL.vocab),
                           embedding_dim=args.embedding_dim,
                           filter_mapping=eval(args.filter_mapping),
                           pretrained_embeddings=TEXT.vocab.vectors,
                           dropout_prob=args.dropout_prob)

if args.cuda:
    if args.data_parallel:
        classifier = torch.nn.DataParallel(classifier,
                                           device_ids=eval(args.device_ids))
    classifier.cuda(device=args.device_id)

criterion = nn.NLLLoss()
optimizer = optim.Adam(classifier.parameters(), args.learning_rate)


# Testing function.
def test(test_set, classifier):
    classifier.eval()

    iterator = data.BucketIterator(dataset=test_set,
                                   batch_size=args.batch_size,
                                   sort_key=lambda x: len(x.text),
                                   device=args.device_id if args.cuda else -1,
                                   train=False)

    loss = 0
    correct = 0
    for batch in iterator:
        output = classifier(batch.text)
        preds = predict(output)

        loss += calc_loss(output, batch.label)
        correct += preds.eq(batch.label.data.view_as(preds)).cpu().sum()

    loss = loss / len(test_set)
    accuracy = correct / len(test_set)

    print(f'test set |',
          f'accuracy: {accuracy * 100:6.2f}% |',
          f'loss: {loss:6.4f} |')


# Training function.
def train(train_set, test_set, classifier, criterion, optimizer, num_epochs):
    classifier.train()

    iterator = data.BucketIterator(dataset=train_set,
                                   batch_size=args.batch_size,
                                   sort_key=lambda x: len(x.text),
                                   device=args.device_id if args.cuda else -1)

    for batch in iterator:
        optimizer.zero_grad()
        loss = criterion(classifier(batch.text), batch.label)
        loss.backward()
        optimizer.step()

        progress, epoch = math.modf(iterator.epoch)

        if iterator.iterations % args.print_every == 0:
            print(f'epoch {int(epoch):2} |',
                  f'progress: {progress * 100:6.2f}% |',
                  f'loss: {loss.data[0]:6.4f} |')

        if progress == 0 and epoch > 0:
            test(test_set, classifier)

        if epoch == args.num_epochs:
            break


if __name__ == '__main__':
    train(train_set, test_set, classifier, criterion, optimizer, args.num_epochs)
