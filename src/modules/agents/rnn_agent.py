import torch.nn as nn
import torch.nn.functional as F
import torch

class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        print("using rnn clamped agent")
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state=None):
        b, a, e = inputs.size()
        if torch.isnan(hidden_state).any():
            print("NAN in hidden")
        x = F.relu(self.fc1(inputs.view(-1, e)), inplace=True)
        if torch.isnan(x).any():
            print("NAN in x")
        if hidden_state is not None:
            hidden_state = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, hidden_state)
        if torch.isnan(h).any():
            print("NAN in h")
        q = self.fc2(h)
        q = torch.clamp(q, -5, 2)
        if torch.isnan(q).any():
            print("NAN in Q1")
        return q.view(b, a, -1), h.view(b, a, -1)
