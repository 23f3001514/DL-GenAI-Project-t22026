# models.py 
# added TextCRNN model (CNN + Bidirectional LSTM) built from scratch
# also added training loop function

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
import wandb


class TextCNN(nn.Module):
    """
    Text CNN model built from scratch.
    Uses multiple filter sizes to capture different n-gram patterns.
    """

    def __init__(self, vocab_size, embedding_dim, num_classes, num_filters, kernel_sizes):
        super(TextCNN, self).__init__()
        self.embedding    = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.conv_kernel2 = nn.Conv1d(embedding_dim, num_filters, kernel_size=2)
        self.conv_kernel3 = nn.Conv1d(embedding_dim, num_filters, kernel_size=3)
        self.conv_kernel4 = nn.Conv1d(embedding_dim, num_filters, kernel_size=4)
        self.dropout      = nn.Dropout(p=0.5)
        self.fc           = nn.Linear(3 * num_filters, num_classes)

    def forward(self, x):
        embedded     = self.embedding(x).permute(0, 2, 1)
        out2         = torch.relu(self.conv_kernel2(embedded))
        out3         = torch.relu(self.conv_kernel3(embedded))
        out4         = torch.relu(self.conv_kernel4(embedded))
        pool2        = torch.max(out2, dim=2)[0]
        pool3        = torch.max(out3, dim=2)[0]
        pool4        = torch.max(out4, dim=2)[0]
        concatenated = torch.cat([pool2, pool3, pool4], dim=1)
        dropped      = self.dropout(concatenated)
        return self.fc(dropped)


class TextCRNN(nn.Module):
    """
    Text CRNN model built from scratch.
    Combines CNN for local patterns with Bidirectional LSTM for sequential patterns.
    
    Why BiLSTM?
        - LSTM remembers long-range dependencies in text
        - Bidirectional means it reads text BOTH forward and backward
        - Forward pass: reads left to right
        - Backward pass: reads right to left
        - Combining both gives full context understanding
    
    Architecture:
        Input -> Embedding -> Conv1D -> BiLSTM -> Dropout -> FC -> Output
    """

    def __init__(self, vocab_size, embedding_dim, num_classes, num_filters, hidden_size):
        super(TextCRNN, self).__init__()

        # embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # CNN layer to extract local features first
        # padding=1 keeps the sequence length the same
        self.conv = nn.Conv1d(
            in_channels  = embedding_dim,
            out_channels = num_filters,
            kernel_size  = 3,
            padding      = 1
        )

        # bidirectional LSTM with 2 layers
        # bidirectional=True means it processes both directions
        # num_layers=2 means we stack 2 LSTM layers on top of each other
        # dropout=0.3 adds dropout between LSTM layers
        self.lstm = nn.LSTM(
            input_size    = num_filters,
            hidden_size   = hidden_size,
            num_layers    = 2,
            batch_first   = True,
            dropout       = 0.3,
            bidirectional = True
        )

        self.dropout = nn.Dropout(p=0.5)

        # hidden_size * 2 because we concatenate forward and backward hidden states
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # embedding
        embedded = self.embedding(x)
        # shape: (batch, seq_len, embed_dim)

        # permute for conv1d
        embedded = embedded.permute(0, 2, 1)
        # shape: (batch, embed_dim, seq_len)

        # CNN to get local features
        conv_output = torch.relu(self.conv(embedded))
        # shape: (batch, num_filters, seq_len)

        # permute back for LSTM input
        conv_output = conv_output.permute(0, 2, 1)
        # shape: (batch, seq_len, num_filters)

        # LSTM - we only use the final hidden states
        lstm_output, (hidden_state, cell_state) = self.lstm(conv_output)

        # hidden_state shape: (num_layers * 2, batch, hidden_size)
        # taking last layer: hidden[-2] = forward, hidden[-1] = backward
        forward_hidden  = hidden_state[-2]
        backward_hidden = hidden_state[-1]

        # concatenate forward and backward hidden states
        combined = torch.cat([forward_hidden, backward_hidden], dim=1)
        # shape: (batch, hidden_size * 2)

        dropped = self.dropout(combined)
        output  = self.fc(dropped)
        return output


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """run one epoch of training and return accuracy."""
    model.train()
    total_loss    = 0
    total_correct = 0
    total_samples = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss    = criterion(outputs, batch_y)
        loss.backward()

        # gradient clipping prevents exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss    += loss.item()
        predictions    = outputs.argmax(dim=1)
        total_correct += (predictions == batch_y).sum().item()
        total_samples += batch_y.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def evaluate_one_epoch(model, val_loader, criterion, device):
    """run one epoch of validation and return accuracy and f1."""
    model.eval()
    total_correct = 0
    total_samples = 0
    all_preds     = []
    all_true      = []

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x  = batch_x.to(device)
            batch_y  = batch_y.to(device)
            outputs  = model(batch_x)
            preds    = outputs.argmax(dim=1)
            total_correct += (preds == batch_y).sum().item()
            total_samples += batch_y.size(0)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_true.extend(batch_y.cpu().numpy().tolist())

    accuracy = total_correct / total_samples
    f1       = f1_score(all_true, all_preds, average='macro')
    return accuracy, f1


def train_model(model, model_name, train_loader, val_loader,
                device, num_epochs=20, learning_rate=0.001,
                wandb_config=None):
    """
    complete training loop with early stopping and wandb logging.
    
    args:
        model        : pytorch model to train
        model_name   : name for saving checkpoint and wandb run
        train_loader : DataLoader for training data
        val_loader   : DataLoader for validation data
        device       : cuda or cpu
        num_epochs   : maximum number of epochs
        learning_rate: initial learning rate
        wandb_config : config dict for wandb
    returns:
        trained model with best weights loaded
    """

    # start wandb run
    try:
        run = wandb.init(
            project = '23f3001514-t22026',
            entity  = '23f3001514-instituition',
            name    = model_name,
            config  = wandb_config or {}
        )
    except Exception as e:
        print(f'wandb init failed: {e}')
        run = None

    # loss function
    criterion = nn.CrossEntropyLoss()

    # adam optimizer with weight decay for regularization
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # reduces learning rate when validation accuracy stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, factor=0.5
    )

    best_val_accuracy = 0
    patience_counter  = 0
    patience_limit    = 5
    save_path         = f'/kaggle/working/{model_name}_best.pt'

    print(f'starting training for {model_name}...')

    for epoch in range(num_epochs):

        # training
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # validation
        val_acc, val_f1 = evaluate_one_epoch(
            model, val_loader, criterion, device
        )

        # update learning rate scheduler
        scheduler.step(val_acc)

        # log to wandb
        if run:
            wandb.log({
                'epoch'        : epoch + 1,
                'train_loss'   : train_loss,
                'train_acc'    : train_acc,
                'val_acc'      : val_acc,
                'val_macro_f1' : val_f1,
                'lr'           : optimizer.param_groups[0]['lr']
            })

        # save best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1

        # early stopping
        if patience_counter >= patience_limit:
            print(f'early stopping at epoch {epoch + 1}')
            break

        # print progress every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f'epoch {epoch+1}/{num_epochs} | '
                  f'train acc: {train_acc:.4f} | '
                  f'val acc: {val_acc:.4f} | '
                  f'val f1: {val_f1:.4f}')

    if run:
        wandb.log({'best_val_acc': best_val_accuracy})
        run.finish()

    # load best weights
    model.load_state_dict(torch.load(save_path))
    print(f'{model_name} training done! best val acc: {best_val_accuracy:.4f}')

    return model


def get_test_probabilities(model, test_loader, device):
    """get softmax probabilities from trained model on test data."""
    model.eval()
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            batch_x = batch[0].to(device)
            outputs = model(batch_x)
            probs   = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_probs, axis=0)