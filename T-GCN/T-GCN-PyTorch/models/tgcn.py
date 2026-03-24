import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.graph_conv import calculate_laplacian_with_self_loop

class TGCNGraphConvolution(nn.Module):
    def __init__(self, adj, num_gru_units: int, output_dim: int, bias: float = 0.0):
        super().__init__()

        self._num_gru_units = num_gru_units
        self._output_dim = output_dim

        self.adjacency = nn.Parameter(torch.FloatTensor(adj))

        self.weights = nn.Parameter(
            torch.FloatTensor(self._num_gru_units + 1, self._output_dim)
        )
        self.biases = nn.Parameter(torch.FloatTensor(self._output_dim))

        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, bias)

    def forward(self, inputs, hidden_state):
        batch_size, num_nodes = inputs.shape

        # ✅ Stable adjacency (VERY IMPORTANT FIX)
        adj = F.softmax(self.adjacency, dim=-1)
        laplacian = calculate_laplacian_with_self_loop(adj)

        inputs = inputs.reshape((batch_size, num_nodes, 1))
        hidden_state = hidden_state.reshape(
            (batch_size, num_nodes, self._num_gru_units)
        )

        concat = torch.cat((inputs, hidden_state), dim=2)
        concat = concat.permute(1, 2, 0).reshape(
            num_nodes, (self._num_gru_units + 1) * batch_size
        )

        a_times_concat = laplacian @ concat
        a_times_concat = a_times_concat.reshape(
            num_nodes, self._num_gru_units + 1, batch_size
        ).permute(2, 0, 1)

        a_times_concat = a_times_concat.reshape(
            batch_size * num_nodes, self._num_gru_units + 1
        )

        outputs = a_times_concat @ self.weights + self.biases
        outputs = outputs.reshape(batch_size, num_nodes * self._output_dim)

        return outputs

class TGCNCell(nn.Module):
    def __init__(self, adj, input_dim: int, hidden_dim: int):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.graph_conv1 = TGCNGraphConvolution(
            adj, hidden_dim, hidden_dim * 2, bias=1.0
        )
        self.graph_conv2 = TGCNGraphConvolution(
            adj, hidden_dim, hidden_dim
        )

    def forward(self, inputs, hidden_state):
        gates = torch.sigmoid(self.graph_conv1(inputs, hidden_state))
        r, u = torch.chunk(gates, 2, dim=1)

        c = torch.tanh(self.graph_conv2(inputs, r * hidden_state))
        new_h = u * hidden_state + (1 - u) * c

        return new_h, new_h

class TGCN(nn.Module):
    def __init__(
        self,
        adj,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2,
        use_layer_norm=True,
        use_attention=True,
        rnn_type="gru",  # "gru", "lstm", "none"
        bidirectional=False,
    ):
        super().__init__()

        self.num_nodes = adj.shape[0]
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_layer_norm = use_layer_norm
        self.use_attention = use_attention
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional

        self.adj = torch.FloatTensor(adj)

        # TGCN Layers
        self.cells = nn.ModuleList([
            TGCNCell(self.adj, self.num_nodes if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])

        # LayerNorm
        if use_layer_norm:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(hidden_dim) for _ in range(num_layers)
            ])

        # Residual projection (FIXED)
        self.residual_proj = nn.Linear(1, hidden_dim)

        self.dropout = nn.Dropout(dropout)

        if use_attention:
            self.attn = nn.Linear(hidden_dim, 1)

        if rnn_type != "none":
            rnn_class = {
                "gru": nn.GRU,
                "lstm": nn.LSTM,
            }[rnn_type]

            self.rnn = rnn_class(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                batch_first=True,
                bidirectional=bidirectional,
            )

            self.rnn_proj = nn.Linear(
                hidden_dim * (2 if bidirectional else 1),
                hidden_dim
            )
        else:
            self.rnn = None

        self.output_network = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, hidden_dim),
        )

    def forward(self, inputs):
        # inputs: (batch, seq_len, num_nodes)
        batch_size, seq_len, num_nodes = inputs.shape

        hidden_states = [
            torch.zeros(batch_size, num_nodes * self.hidden_dim, device=inputs.device)
            for _ in range(self.num_layers)
        ]

        time_outputs = []

        for t in range(seq_len):
            x = inputs[:, t, :]
            residual = x

            for i, cell in enumerate(self.cells):
                out, h = cell(x, hidden_states[i])
                hidden_states[i] = h

                out = out.reshape(batch_size, num_nodes, self.hidden_dim)

                # ✅ Fixed residual
                res = self.residual_proj(residual.unsqueeze(-1))
                out = out + res

                if self.use_layer_norm:
                    out = self.layer_norms[i](out)

                out = self.dropout(out)

                x = out.mean(dim=2)
                residual = x

            time_outputs.append(out)

        time_outputs = torch.stack(time_outputs, dim=1)

        if self.use_attention:
            scores = self.attn(time_outputs)
            weights = F.softmax(scores, dim=1)
            out = (time_outputs * weights).sum(dim=1)
        else:
            out = time_outputs[:, -1]

        if self.rnn is not None:
            rnn_in = time_outputs.permute(0, 2, 1, 3)  # (B, N, T, H)
            rnn_in = rnn_in.reshape(batch_size * num_nodes, seq_len, self.hidden_dim)

            rnn_out, _ = self.rnn(rnn_in)
            last = rnn_out[:, -1]

            last = self.rnn_proj(last)
            out = last.reshape(batch_size, num_nodes, self.hidden_dim)

        # Output
        out = self.output_network(out)

        return out

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--hidden_dim", type=int, default=128)
        parser.add_argument("--num_layers", type=int, default=2)
        parser.add_argument("--dropout", type=float, default=0.2)
        parser.add_argument("--use_layer_norm", type=bool, default=True)
        parser.add_argument("--use_attention", type=bool, default=True)
        parser.add_argument("--rnn_type", type=str, default="gru", choices=("gru", "lstm", "none"))
        parser.add_argument("--bidirectional", action="store_true")

        return parser

    @property
    def hyperparameters(self):
        return {
            "num_nodes": self.num_nodes,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout.p,
            "use_layer_norm": self.use_layer_norm,
            "use_attention": self.use_attention,
            "rnn_type": self.rnn_type,
            "bidirectional": self.bidirectional,
        }