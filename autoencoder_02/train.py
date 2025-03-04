import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from tqdm import tqdm
from dataloader import MyDataloader
from model import Autoencoder
from parse import parse_args
from utils import plot_representation, show_images

# Get configuration
args = parse_args()
batch_size = args.batch_size
epochs = args.epochs
encoding_dim_input = args.encoding_dim_input
encoding_dim_output = args.encoding_dim_output

# Load data
train_loader, val_loader, test_loader = MyDataloader(batch_size=batch_size)

# Intiate Model
model = Autoencoder(encoding_dim_input, encoding_dim_output)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Train process
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())
writer = SummaryWriter()
best_loss = np.inf
best_model_path = 'best_model.pt'

for epoch in range(epochs):
    # training 
    model.train()  
    running_loss = 0.0
    
    for images, _ in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} [Train]'):
        images = images.to(device)
        # print(images[0])
        outputs = model(images)
        loss = criterion(outputs, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    train_loss = running_loss / len(train_loader)
    writer.add_scalar('Loss/Train Loss', train_loss, epoch)
    print(f'Train Loss: {train_loss:.4f}')
    

    # Validation
    model.eval()
    val_running_loss = 0.0

    with torch.no_grad(): 
        for images, _ in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{epochs} [Val]'):
            images = images.to(device)
            outputs = model(images)
            loss = criterion(outputs, images)
            val_running_loss += loss.item()

    val_loss = val_running_loss / len(val_loader)
    writer.add_scalar('Loss/Validation Loss', val_loss, epoch)
    print(f'Validation Loss: {val_loss:.4f}')

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), best_model_path)

writer.close()   


# ===================Prediction========================

checkpoint_path = f'best_model.pt'
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model state dictionary
model = Autoencoder(encoding_dim_input, encoding_dim_output)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device) 
model.eval()

# out_file = 'prediction.csv'

total_test = 0
test_running_loss = 0.0

with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)
        outputs = model(images)
        loss = criterion(outputs, images)
        test_running_loss += loss.item()

test_loss = test_running_loss / len(test_loader)
print(f'Test Loss: {test_loss:.4f}')

# ===================Visualize========================
test_dataset = datasets.MNIST(root='./data/mnist', train=False, download=True,
                                transform=transforms.ToTensor())
x_test = test_dataset.data.float() / 255.0
y_test = test_dataset.targets
x_test = x_test.reshape(x_test.shape[0], -1).numpy()


with torch.no_grad():
    x_test_tensor = torch.FloatTensor(x_test).to(device)
    encode_images = model.encode(x_test_tensor).cpu().numpy()
    decode_images = model(x_test_tensor).cpu().numpy()

plot_representation(encode_images, y_test)
show_images(decode_images, x_test)











