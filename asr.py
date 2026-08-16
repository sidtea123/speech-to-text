import torch
import torch.nn as nn
from data_parser import *

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=(2, 2), padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=(1, 1), padding=1),
            nn.BatchNorm2d(32),
            nn.GELU()
        )

    def forward(self, x):
        return self.network(x)

class STTModel(nn.Module):
    def __init__(self, n_mels=64, hidden_dim=512, num_characters=29):
        super().__init__()
        cnn_out_mels = n_mels // 2
        rnn_input_dim = 32 * cnn_out_mels
        
        self.lstm = nn.LSTM(
            input_size=rnn_input_dim,
            hidden_size=hidden_dim,
            num_layers=3,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )

        self.linear_classifier = nn.Linear(hidden_dim * 2, num_characters)
        self.cnn = CNN()

    def forward(self, x):
        x = self.cnn(x)
        
        # batch_size, channels, time_reduced, mels_reduced = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.flatten(start_dim=2, end_dim=3)
        x, _ = self.lstm(x) 
        
        logits = self.linear_classifier(x) 
        
        return torch.log_softmax(logits, dim=-1)

MODEL_URL = 'speech_data/stt_model.pt'
BATCH_SIZE = 131

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = STTModel().to(device, non_blocking=True)

    images, labels = read_files()
    print(images.shape)
    print(labels.shape)
    data_loader = generate_dataset_and_loader(images, labels, BATCH_SIZE)

    epochs = 1
    lr = 0.001

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_function = nn.CTCLoss()

    print(f'\nfinished preparing model, now running on {device}...\n')

    for epoch in range(1, epochs + 1):
        mismatches = 0
        total_loss = 0
        batch_num = 0
        for x, y in data_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            output_lengths = torch.tensor(np.full(BATCH_SIZE, 401), dtype=torch.int64).to(device)
            target_lengths = torch.tensor(np.full(BATCH_SIZE, 186), dtype=torch.int64).to(device)

            y_hat = model(x)
            output = y_hat.permute(1, 0, 2)
            loss = loss_function(output, y, output_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # mismatches += (torch.argmax(y_hat, dim=1) != y).sum().item()
            total_loss += loss.item()
            batch_num += 1

            if batch_num % 5 == 0:
                print(f'finished {batch_num} / {NUM_FILES / BATCH_SIZE} for epoch {epoch}')
                print(f'current loss: {total_loss / (batch_num * BATCH_SIZE)}')

        print(f'\nProgress at epoch {epoch}:')
        print(f'loss: {total_loss / NUM_FILES}')
        torch.save(model.state_dict(), MODEL_URL)
        print(f'saved current model state to {MODEL_URL}.\n')