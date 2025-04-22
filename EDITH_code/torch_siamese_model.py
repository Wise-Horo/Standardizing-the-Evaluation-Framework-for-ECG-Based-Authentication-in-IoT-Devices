import torch
import torch.nn as nn
class SiameseModel(nn.Module):
    def __init__(self, feat_len=128):
        super().__init__()

        # Input is 'L2'
        self.path1 = nn.Sequential(
            nn.Linear(in_features=feat_len, out_features=64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.25),
            nn.ReLU()
        )

        # Input is 'prod'
        self.path2 = nn.Sequential(
            nn.Linear(in_features=feat_len, out_features=64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.25),
            nn.ReLU()
        )

        # Input is 'combine'
        self.path3 = nn.Sequential(
            nn.Linear(in_features=feat_len*2, out_features=64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.25),
            nn.ReLU()
        )

        # Input is 'paths', output of concatenate(path1, path2, path3)
        self.top = nn.Sequential(
            nn.Linear(in_features=64*3, out_features=256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.25),
            nn.ReLU()
        )

        self.out = nn.Linear(in_features=256, out_features=1)
    

    def forward(self, x1, x2):

        diff = torch.subtract(x1, x2)
        L2 = torch.multiply(diff, diff)
        prod = torch.multiply(x1, x2)
        combine = torch.cat([L2, prod], axis=1)

        path1 = self.path1(L2)
        path2 = self.path2(prod)
        path3 = self.path3(combine)

        concat = torch.cat([path1, path2, path3], axis=1)
        out = self.top(concat)
        out = self.out(out)
        out = out.flatten()
       
        return out 