# models.py
# deep learning model definitions for Smart MCQ Solver
# this file contains TextCNN model built completely from scratch using PyTorch

import torch
import torch.nn as nn


class TextCNN(nn.Module):
    """
    Text Classification using Convolutional Neural Network.
    Built completely from scratch.
    
    Architecture:
        Input -> Embedding -> Conv1D (3 filter sizes) -> MaxPool -> Dropout -> FC -> Output
    
    Why CNN for text?
        - Different filter sizes capture different length patterns
        - kernel_size=2 captures 2-word patterns (bigrams)
        - kernel_size=3 captures 3-word patterns (trigrams)
        - kernel_size=4 captures 4-word patterns
        - Global max pooling keeps the most important feature from each filter
        - This is better than just one filter size
    
    Args:
        vocab_size    : total number of unique words in vocabulary
        embedding_dim : size of each word embedding vector (128)
        num_classes   : number of output classes (5 for A-E)
        num_filters   : number of filters per kernel size (128)
        kernel_sizes  : list of filter sizes to use (default [2, 3, 4])
    """

    def __init__(self, vocab_size, embedding_dim, num_classes, num_filters, kernel_sizes):
        super(TextCNN, self).__init__()

        # embedding layer
        # converts word indices to dense vector representations
        # padding_idx=0 means the <PAD> token gets zero embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # convolutional layers with different kernel sizes
        # each captures different length patterns in text
        self.conv_kernel2 = nn.Conv1d(
            in_channels  = embedding_dim,
            out_channels = num_filters,
            kernel_size  = 2
        )
        self.conv_kernel3 = nn.Conv1d(
            in_channels  = embedding_dim,
            out_channels = num_filters,
            kernel_size  = 3
        )
        self.conv_kernel4 = nn.Conv1d(
            in_channels  = embedding_dim,
            out_channels = num_filters,
            kernel_size  = 4
        )

        # dropout for regularization
        # randomly sets 50% of neurons to zero during training
        # this prevents the model from overfitting
        self.dropout = nn.Dropout(p=0.5)

        # final fully connected layer
        # 3 * num_filters because we have 3 different kernel sizes
        # maps from feature space to number of classes
        self.fc = nn.Linear(3 * num_filters, num_classes)

    def forward(self, x):
        """
        forward pass through the network.
        
        args:
            x: input tensor of shape (batch_size, sequence_length)
               contains word indices
        returns:
            logits tensor of shape (batch_size, num_classes)
        """

        # step 1: get word embeddings
        # shape changes from (batch, seq_len) to (batch, seq_len, embed_dim)
        embedded = self.embedding(x)

        # step 2: permute dimensions for Conv1D
        # Conv1D expects (batch, channels, length)
        # so we change from (batch, seq_len, embed_dim) to (batch, embed_dim, seq_len)
        embedded = embedded.permute(0, 2, 1)

        # step 3: apply each convolutional filter with relu activation
        # relu removes negative values (keeps only positive features)
        out_kernel2 = torch.relu(self.conv_kernel2(embedded))
        out_kernel3 = torch.relu(self.conv_kernel3(embedded))
        out_kernel4 = torch.relu(self.conv_kernel4(embedded))

        # step 4: global max pooling
        # takes the maximum value across the entire sequence for each filter
        # this gives us the most important feature regardless of position
        # dim=2 means we pool across the sequence length dimension
        pool_kernel2 = torch.max(out_kernel2, dim=2)[0]
        pool_kernel3 = torch.max(out_kernel3, dim=2)[0]
        pool_kernel4 = torch.max(out_kernel4, dim=2)[0]

        # step 5: concatenate all pooled outputs
        # shape: (batch_size, 3 * num_filters)
        concatenated = torch.cat([pool_kernel2, pool_kernel3, pool_kernel4], dim=1)

        # step 6: apply dropout
        dropped = self.dropout(concatenated)

        # step 7: final classification
        # shape: (batch_size, num_classes)
        output = self.fc(dropped)

        return output


def create_textcnn(vocab_size, num_classes=5):
    """
    helper function to create a TextCNN model with default settings.
    
    args:
        vocab_size  : size of vocabulary
        num_classes : number of output classes (default 5)
    returns:
        TextCNN model instance
    """
    model = TextCNN(
        vocab_size    = vocab_size,
        embedding_dim = 128,
        num_classes   = num_classes,
        num_filters   = 128,
        kernel_sizes  = [2, 3, 4]
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f'TextCNN created with {total_params:,} total parameters')

    return model


if __name__ == '__main__':
    # quick test to make sure model works
    vocab_size  = 10000
    batch_size  = 4
    seq_length  = 256

    model  = create_textcnn(vocab_size)
    x_test = torch.randint(0, vocab_size, (batch_size, seq_length))
    output = model(x_test)

    print(f'input shape  : {x_test.shape}')
    print(f'output shape : {output.shape}')
    print('TextCNN forward pass working correctly!')