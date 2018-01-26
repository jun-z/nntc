import torch
import torch.nn as nn


class CNNClassifier(nn.Module):
    def __init__(self,
                 vocab_size,
                 label_size,
                 embedding_dim,
                 filter_mapping,
                 dropout_prob=.5):

        super(CNNClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

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
