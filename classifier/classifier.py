import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, input_dim=896*1, hidden_dim=1024, dropout=0.1, task='len', num_token=None):
        super().__init__()
        if task == 'token':
            assert num_token is not None, "When task='token', num_token must be provided"
            out_dim = num_token
        else:  # task == 'len' or 'correctness' or 'logicality'
            out_dim = 1

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, out_dim)
        )

    def forward(self, x):
        logits = self.net(x)
        return logits