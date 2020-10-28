import torch

def euclidean_metrix(X, centers):
    '''
    calculate the negative l2 distance between feature X and proto centers
    :param X: encoding feature [N, dim]
    :param centers: proto centers [C, dim]
    :return: logits (scores)
    '''
    m = X.size(0)
    n = centers.size(0)

    X = torch.unsqueeze(X, 1)
    X = X.expand(m, n, -1)

    centers = torch.unsqueeze(centers, 0)
    centers = centers.expand(m, n, -1)

    logits = -((X - centers) ** 2).sum(dim=2)

    return logits
