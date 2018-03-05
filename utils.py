import torch.nn.functional as F


def pad_shorties(x, min_len, pad_token='<pad>'):
    if len(x) < min_len:
        return x + [pad_token] * (min_len - len(x))
    else:
        return x


def calc_loss(output, labels):
    return F.nll_loss(output, labels, size_average=False).data[0]


def predict(output):
    return output.data.max(1, keepdim=True)[1]
