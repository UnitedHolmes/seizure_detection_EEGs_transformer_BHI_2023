"""
Adapted from: Continuous Seizure Detection Based on Transformer and Long-Term iEEG
"""

import math
import os
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import copy
import time
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
from sklearn.metrics import f1_score

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class CNNModule(nn.Module):
    def __init__(self, CNN_last_out_channels=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 4),
                               stride=(1,2), padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 4),
                               stride=(1,2), padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=CNN_last_out_channels, kernel_size=(1, 4),
                               stride=(1,2), padding=0)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        # self.pooling = nn.AvgPool2d((1, ))

    def forward(self, x):
        # input x shape: (batch_size, num_channels, seq_len) = (batch_size, 22, 1000)        
        if debug_mode_flag: print('Shape of x in the beginning of CNNModule: ', x.shape)
        x = torch.unsqueeze(x, 1)
        if debug_mode_flag: print('Shape of x before conv1: ', x.shape)
        x = self.conv1(x)
        if debug_mode_flag: print('Shape of x after conv1: ', x.shape)
        x = self.relu(x)
        x = self.bn1(x)
        if debug_mode_flag: print('Shape of x before conv2: ', x.shape)
        x = self.conv2(x)
        if debug_mode_flag: print('Shape of x after conv2: ', x.shape)
        x = self.relu(x)
        x = self.bn2(x)
        if debug_mode_flag: print('Shape of x before conv3: ', x.shape)
        x = self.conv3(x)
        if debug_mode_flag: print('Shape of x after conv3: ', x.shape)
        x = self.relu(x)
        out = torch.mean(x, 3) # output shape is (batch size, cnn_filters=128, eeg_channels=22)
        if debug_mode_flag: print('Shape of x after average pooling: ', out.shape)
        
        return out

class CNNTransformer(nn.Module):
    def __init__(self, nclasses, nhead, num_layers, dim_feedforward, CNN_last_out_channels=128, dropout=0.1):
        super(CNNTransformer, self).__init__()
        self.cnn = CNNModule(CNN_last_out_channels)
        encoder_layers = TransformerEncoderLayer(d_model=CNN_last_out_channels, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer = TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(CNN_last_out_channels, nclasses)  
        self.pos_encoder = PositionalEncoding(CNN_last_out_channels, dropout)

    def forward(self, x):
        
        # input x shape: (batch_size, num_channels, seq_len) = (batch_size, 22, 1000)
        x = self.cnn(x) # output shape is (batch size, cnn_filters=128, eeg_channels=22)
        if debug_mode_flag: print('Shape of x after CNN Module: ', x.shape)
        
        batch_size, seq_len, num_channels = x.shape ## cnn_filters == seq_len
        x = torch.cat((torch.zeros((batch_size, seq_len, 1), requires_grad=True).to(device), x), 2)        
        # output shape: (batch size, cnn_filters=128, eeg_channels=22+1)
        x = x.permute(2, 0, 1) # ouptut shape: (channels+1, batch_size, seq_len). cnn_filters (or seq_len) is seen as the embedding size
        if debug_mode_flag: print('Shape of x before Transformer: ', x.shape)
        if flag_positional_encoding:
            # x = x * math.sqrt(self.d_model)
            x = x * math.sqrt(seq_len)
            x = self.pos_encoder(x) ## output matrix shape: (num_channels+1, batch_size, seq_len)
        if debug_mode_flag: print('Positional Encoding Done!')
        if debug_mode_flag: print('Shape of x after positional encoding: ', x.shape)
        x = self.transformer(x) # ouptut shape: (channels+1, batch_size, seq_len)
        out = x[0,:,:].reshape(batch_size, -1)
        res = self.fc(out)
        return res

binary_classifier_flag = True
# binary_classifier_flag = False
debug_mode_flag = False
balanced_training_flag = True
flag_positional_encoding = True
batch_size = 2048  # Set your desired batch size

### Transformer model hyperparameters
segment_interval = 4
resampleFS = 250
sequence_length = resampleFS * segment_interval
num_eeg_channels=22

if binary_classifier_flag:
    nclasses = 2
else:
    nclasses = 4

if binary_classifier_flag:
    seizure_types = ['bckg', 'seizure']
    if balanced_training_flag:
        data_root = os.path.join('/datadrive', 'TUSZ', 'new_3_balanced_TUSZ_processed_binary_individual_segments', 'segment_interval_'+str(segment_interval)+'_sec')
    else:
        data_root = os.path.join('/datadrive', 'TUSZ', 'new_3_TUSZ_processed_binary_individual_segments', 'segment_interval_'+str(segment_interval)+'_sec')
else:
    seizure_types = ['fnsz', 'gnsz', 'cpsz', 'bckg']
    if balanced_training_flag:
        data_root = os.path.join('/datadrive', 'TUSZ', 'new_3_balanced_TUSZ_processed_multiclass_individual_segments', 'segment_interval_'+str(segment_interval)+'_sec')
    else:
        data_root = os.path.join('/datadrive', 'TUSZ', 'new_3_TUSZ_processed_multiclass_individual_segments', 'segment_interval_'+str(segment_interval)+'_sec')

dim_feedforward = 256  # dimension of the feedforward network model in nn.TransformerEncoder
num_layers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 4  # number of heads in nn.MultiheadAttention
dropout = 0.1  # dropout probability

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# model = CNNModule().to(device)

model = CNNTransformer(nclasses=nclasses, nhead=nhead, num_layers=num_layers, dim_feedforward=dim_feedforward).to(device)

model_stats = summary(model, input_size=(batch_size, num_eeg_channels, sequence_length), verbose=0)
print(model_stats)

# ## 1. Create a dataset
# PyTorch provides an abstract class representing a Dataset. Your custom dataset should inherit Dataset and override the following methods:
# __len__ so that len(dataset) returns the size of the dataset.
# __getitem__ to support the indexing such that dataset[i] can be used to get i-th sample.
# An alternate way is to return sequences from the numpy file in the __getitem__ method. This way, each call to the dataloader would return a sequence (or a batch of sequences) instead of a whole numpy file.
# This dataset will load all numpy files and store each sequence separately along with its label. Please note that this approach may consume more memory because it loads all the data at once. If your dataset is too large to fit into memory, you may need to use a different approach, like loading only a part of the data or loading each file on demand during the training.

class NumpyDataset(Dataset):
    def __init__(self, root_dir):#, sequence_length=1000):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.classes = seizure_types #os.listdir(root_dir)
        self.data = []
        for class_index, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            class_files = os.listdir(class_dir)
            for file_name in class_files:
                data = np.load(os.path.join(class_dir, file_name))
                n_samples = data.shape[0]
                for i in range(n_samples):
                    self.data.append((data[i], class_index))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data, label = self.data[idx]
        return data, label

# ## 2. Define a DataLoader
# DataLoader combines a dataset and a sampler, and provides an iterable over the given dataset. It also provides multi-process data loading with the num_workers argument.

train_dir = os.path.join(data_root, 'train')  # Specify the path to your root directory
train_dataset = NumpyDataset(train_dir)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)


val_dir = os.path.join(data_root, 'val')  # Specify the path to your root directory
val_dataset = NumpyDataset(val_dir)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

test_dir = os.path.join(data_root, 'test')  # Specify the path to your root directory
test_dataset = NumpyDataset(test_dir)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# ## 3. Define and Train the model
# Now, let's train our model:


if binary_classifier_flag:
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.NLLLoss()
else:
    # criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()

lr = 0.0001  # learning rate
# optimizer = torch.optim.SGD(model.parameters(), lr=lr)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
STEPLR_period = 20.0
STEPLR_gamma = 0.1
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, STEPLR_period, gamma=STEPLR_gamma)

best_val_loss = float('inf')
epochs = 1000
best_model = None

loss_train_array, loss_val_array = [], []

starting_time = datetime.now()

dt_string = starting_time.strftime("%Y-%m-%d-%H-%M")
print("Starting date and time =", dt_string)

root_path_keywords = 'results_CNN_transformer_2023_someone_else_model'

if not os.path.exists(root_path_keywords):
    os.mkdir(root_path_keywords)

if binary_classifier_flag:
    results_save_folder = os.path.join(root_path_keywords, 'binary_experiment_' + dt_string)
else:
    results_save_folder = os.path.join(root_path_keywords, 'multi_class_experiment_' + dt_string)
if not os.path.exists(results_save_folder):
    os.mkdir(results_save_folder)

model_save_path = os.path.join(results_save_folder, 'best_model.pt')

from torchinfo import summary
model_stats = summary(model, input_size=(batch_size, num_eeg_channels, sequence_length), verbose=0)

model_summary_save_path = os.path.join(results_save_folder, 'model_summary.txt')
    
with open(model_summary_save_path, 'w') as txtFile:
    for value in str(model_stats):
        txtFile.write(str(value))

def train(model: nn.Module, train_dataloader):
    model.train()  # turn on train mode
    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data
        # print(inputs.shape)
        # print(labels)
        inputs, labels = inputs.float(), labels.long()  # ensure correct data type
        inputs, labels = inputs.to(device), labels.to(device)  # if you're using GPU

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        running_loss += loss.item()
        
    return running_loss / len(train_dataloader)

def evaluate(model: nn.Module, val_dataloader):
    model.eval()  # turn on evaluation mode
    val_running_loss = 0.
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for i, data in enumerate(val_dataloader, 0):
            inputs, labels = data
            inputs, labels = inputs.float(), labels.long()  # ensure correct data type
            inputs, labels = inputs.to(device), labels.to(device)  # if you're using GPU

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    return val_running_loss / len(val_dataloader), val_correct / val_total

def evaluate_test(model: nn.Module, test_dataloader):
    model.eval()  # turn on evaluation mode
    test_running_loss = 0.
    test_correct = 0
    test_total = 0
    total_predicted = []
    total_labels = []
    with torch.no_grad():
        for i, data in enumerate(test_dataloader, 0):
            inputs, labels = data
            inputs, labels = inputs.float(), labels.long()  # ensure correct data type
            inputs, labels = inputs.to(device), labels.to(device)  # if you're using GPU

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            total_predicted += predicted.to('cpu').numpy().tolist()
            total_labels += labels.to('cpu').numpy().tolist()

    return test_running_loss / len(test_dataloader), test_correct / test_total, total_predicted, total_labels


# Define your early stopping parameters
patience = 50
min_delta = 0.01

# These variables are for tracking the best score and patience counter
best_score = float('inf')
best_epoch = None

no_improvement_counter = 0
# test_macro_f1_score = []

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train_loss = train(model, train_dataloader)
    val_loss, val_acc = evaluate(model, val_dataloader)

    loss_train_array.append(train_loss)
    loss_val_array.append(val_loss)

    elapsed = time.time() - epoch_start_time
    print('-' * 89)
    print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
          f'| training loss {train_loss:5.3f} | '
          f'valid loss {val_loss:5.3f} | valid acc {val_acc:8.3f} ')
    print('-' * 89)

    scheduler.step()

    if best_score > val_loss + min_delta:
        best_score = val_loss
        best_model = copy.deepcopy(model)
        best_epoch = epoch
        no_improvement_counter = 0
    else:
        no_improvement_counter += 1  

    if no_improvement_counter >= patience:
        print("Early stopping at epoch:", epoch)
        torch.save(best_model.state_dict(), model_save_path)
        early_stop_epoch = epoch
        break
    
    if epoch == epochs:
        torch.save(best_model.state_dict(), model_save_path)


def plot_train_val_loss(loss_train_array, loss_val_array, results_save_folder):

    plt.figure()
    plt.clf()
    plt.title("Training and Validation Loss")
    plt.plot(loss_val_array,label="val")
    plt.plot(loss_train_array,label="train")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(results_save_folder, 'Train_loss_curve.png'))

plot_train_val_loss(loss_train_array, loss_val_array, results_save_folder)

test_loss, test_acc, y_pred, y_test = evaluate_test(best_model, test_dataloader)
y_test = np.array(y_test).reshape(-1)
y_pred = np.array(y_pred).reshape(-1)
print(y_test.shape)
print(y_pred.shape)
if len(y_pred) < len(y_test):
    y_test = np.delete(y_test, range(len(y_pred),len(y_test)))
# test_ppl = math.exp(test_loss)
print('=' * 89)
print(f'| End of training | test loss {test_loss:5.2f} | '
      f'test acc {test_acc:8.2f}')
print('=' * 89)


### Save the confusion matrix
cm = metrics.confusion_matrix(y_test, y_pred)
if binary_classifier_flag:
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=['Non-Seizure', 'Seizure'])
else:
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=['fnsz', 'gnsz', 'cpsz', 'bckg'])
cm_display.plot(cmap=plt.cm.Blues)
plt.savefig(os.path.join(results_save_folder, 'confusion_matrix.png'))


# Print the precision and recall, among other metrics
print(metrics.classification_report(y_test, y_pred, digits=3))


results_save_path = os.path.join(results_save_folder, 'output_results_' + str(segment_interval) + '_sec_segments.txt')
    
with open(results_save_path, 'w') as txtFile:
    for value in metrics.classification_report(y_test, y_pred, digits=3):
        txtFile.write(str(value))
        # txtFile.write('\n')

hyperparameters_save_path = os.path.join(results_save_folder, 'hyperparameters.txt')

with open(hyperparameters_save_path, 'w') as txtFile:
    txtFile.write('Segment interval: ' + str(segment_interval))
    txtFile.write('\n')
    txtFile.write('If binary classification: ' + str(binary_classifier_flag))
    txtFile.write('\n')
    txtFile.write('If Balanced Training: ' + str(balanced_training_flag))
    txtFile.write('\n')
    txtFile.write('If Positional Encoding: ' + str(flag_positional_encoding))
    txtFile.write('\n')
    txtFile.write('Training epochs: ' + str(epochs))
    txtFile.write('\n')
    txtFile.write('Best model epoch: ' + str(best_epoch))
    txtFile.write('\n')        
    txtFile.write('Learning rate: ' + str(lr))
    txtFile.write('\n')
    txtFile.write('Decaying period: ' + str(STEPLR_period))
    txtFile.write('\n')
    txtFile.write('Decaying gamma rate: ' + str(STEPLR_gamma))
    txtFile.write('\n')
    txtFile.write('Time taken to run: ' + str(datetime.now() - starting_time))
    txtFile.write('\n')
    txtFile.write('############ Hyperparameters for Transformer Model')
    txtFile.write('\n')
    txtFile.write('Dimension of the feedforward network model in nn.TransformerEncoder: ' + str(dim_feedforward))
    txtFile.write('\n')
    txtFile.write('Number of nn.TransformerEncoderLayer in nn.TransformerEncoder: ' + str(num_layers))
    txtFile.write('\n')
    txtFile.write('Number of heads in nn.MultiheadAttention: ' + str(nhead))
    txtFile.write('\n')
    txtFile.write('Training data root: ' + str(data_root))


