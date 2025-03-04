import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, encoding_dim), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 256), nn.ReLU(),
            nn.Linear(256, input_dim), nn.ReLU()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = decoded.view(x.size(0), 1, 28, 28)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)
    

if __name__ == "__main__":
    model = Autoencoder(784, 10)
    print(model)

    dummy_input = torch.randn(1, 784)  
    output = model(dummy_input)
    print("Output shape:", output.shape)