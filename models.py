import torch
import torch.nn as nn
import torch.autograd as autograd


class CNNClassifier(nn.Module):
    def __init__(self,
                 vocab_size,
                 label_size,
                 embedding_dim,
                 filter_mapping,
                 pretrained_embeddings=None,
                 dropout_prob=.5):

        super(CNNClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)

        self.convs = nn.ModuleList()
        for filter_size, num_filters in filter_mapping.items():
            self.convs.append(nn.Conv2d(1, num_filters, (filter_size, embedding_dim)))

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.linear = nn.Linear(sum(filter_mapping.values()), label_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_):
        embeddings = self.embedding(input_).unsqueeze(1)

        features = []
        for conv in self.convs:
            feature_map = self.relu(conv(embeddings)).squeeze(-1)
            feature, _ = feature_map.max(-1)
            features.append(feature)

        features = self.dropout(torch.cat(features, 1))
        return self.softmax(self.linear(features))


class LSTMAttentionClassifier(nn.Module):
    def __init__(self,
                 vocab_size,
                 label_size,
                 embedding_dim,
                 hidden_dim,
                 num_layers=1):
        super(LSTMAttentionClassifier, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers,
                            batch_first=True,
                            bidirectional=True)

        self.linear = nn.Linear(hidden_dim * 2, 1)
        self.relu = nn.ReLU()

        self.proj = nn.Linear(embedding_dim, label_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sequence):
        embeddings = self.embedding(sequence)

        hidden_size = (self.num_layers * 2,
                       sequence.size()[0],
                       self.hidden_dim)

        self.hidden = (autograd.Variable(torch.randn(*hidden_size)),
                       autograd.Variable(torch.randn(*hidden_size)))

        if sequence.is_cuda:
            self.hidden = [h.cuda(sequence.get_device()) for h in self.hidden]

        encodings, self.hidden = self.lstm(embeddings, self.hidden)

        attentions = self.relu(self.linear(encodings))
        encoding = torch.bmm(attentions.transpose(1, 2), embeddings).squeeze(1)

        return self.softmax(self.proj(encoding))
