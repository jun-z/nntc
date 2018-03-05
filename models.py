import torch
import torch.nn as nn
import torch.autograd as autograd


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


class CapsClassifier(nn.Module):
    def __init__(self,
                 vocab_size,
                 label_size,
                 embedding_dim,
                 hidden_dim,
                 filter_mapping,
                 num_iterations,
                 dropout_prob=.5):

        super(CapsClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.convs = nn.ModuleList()
        self.projs = nn.ModuleList()
        for filter_size, num_filters in filter_mapping.items():
            self.convs.append(nn.Conv2d(1, num_filters, (filter_size, embedding_dim)))
            self.projs.append(nn.Linear(num_filters, hidden_dim))

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.linear = nn.Linear(hidden_dim, label_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)

        self.num_iterations = num_iterations

    def forward(self, input_):
        embeddings = self.embedding(input_).unsqueeze(1)

        capsule_groups = []
        for conv, proj in zip(self.convs, self.projs):
            capsule_group = self.relu(conv(embeddings)).squeeze(-1).transpose(1, 2)
            capsule_group = self.tanh(proj(capsule_group))
            capsule_groups.append(capsule_group)

        capsules = torch.cat(capsule_groups, 1)

        log_priors = autograd.Variable(torch.zeros(capsules.size()[:2]))

        if capsules.is_cuda:
            log_priors = log_priors.cuda(capsules.get_device())

        for i in range(self.num_iterations):
            weights = self.softmax(log_priors).unsqueeze(1)
            features = self.tanh(torch.bmm(weights, capsules).transpose(1, 2))
            log_priors = log_priors + torch.bmm(capsules, features).squeeze(-1)

        features = self.dropout(features.squeeze(-1))
        return self.logsoftmax(self.linear(features))
