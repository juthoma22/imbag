from torch import nn


class GeographicalClassifier(nn.Module):
    def __init__(self, embedding_dim=768, hidden_dims=[512, 256, 128], num_classes=1093, dropout_rate=0.5):
        super(GeographicalClassifier, self).__init__()
        
        # Dynamically create layers based on the hidden_dims list
        self.layers = nn.ModuleList()
        input_dim = embedding_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            self.layers.append(nn.ReLU(inplace=True))
            self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_dim  # Set next layer's input size
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], num_classes)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        # Apply output layer
        x = self.output_layer(x)
        return x
