from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F


def test_dataset(dataset):

    batch_size = [1, 2, 3]
    for size in batch_size:
        loader = DataLoader(dataset, batch_size=size)
        count = 0
        for x in loader:
            count = 1 
        if size == 1:
            assert count == len(dataset)
        assert count == len(loader)
    
    print("pass!")