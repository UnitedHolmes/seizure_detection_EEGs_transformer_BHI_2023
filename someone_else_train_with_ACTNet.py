## Adapted from https://github.com/Altaheri/EEG-ATCNet

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


class CNNBlock(nn.Module):
    def __init__(self, F1=4, kernLength=64, D=2, Chans=22, dropout=0.5, eegn_poolSize=8):
        super(CNNBlock, self).__init__()

        self.F1 = F1
        self.kernLength = kernLength
        self.D = D
        F2 = F1*D
        self.Chans = Chans
        self.dropout = nn.Dropout(dropout)

        self.block1 = nn.Conv2d(1, F1, (1, kernLength), padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.block2 = nn.Conv2d(F1, F1*D, (Chans, 1), groups=F1, padding='valid', bias=False)
        self.bn2 = nn.BatchNorm2d(F1*D)
        self.elu = nn.ELU()
        self.avg_pool1 = nn.AvgPool2d((1, 8))
        self.block3 = nn.Conv2d(F1*D, F2, (1, 16), padding='same', groups=F1*D, bias=False)
        self.bn3 = nn.BatchNorm2d(F1*D)
        self.avg_pool2 = nn.AvgPool2d((1, eegn_poolSize))

    def forward(self, x):
        # x.shape = (B, 1, C, L)

        x = self.block1(x)
        # x.shape = (B, F1, C, L)
        # print('Shape of x after block1: ', x.shape)

        x = self.bn1(x)
        x = self.block2(x)
        # x.shape = (B, F1*D, 1, L)
        # print('Shape of x after block2: ', x.shape)

        x = self.bn2(x)
        x = self.elu(x)
        x = self.avg_pool1(x)
        x = self.dropout(x)
        # x.shape = (B, F1*D, 1, L/8)
        # print('Shape of x before block3: ', x.shape)
        x = self.block3(x)
        # x.shape = (B, F1*D, 1, L/8)
        # print('Shape of x after block3: ', x.shape)

        x = self.bn3(x)
        x = self.elu(x)
        x = self.avg_pool2(x)
        x = self.dropout(x)
        # x.shape = (B, F1*D, 1, L/64)

        return x

class TCNBlock(nn.Module):
    def __init__(self, input_dimension, tcn_kernel_size, tcn_filters, dropout):
        super(TCNBlock, self).__init__()

        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)

        self.conv1 = nn.Conv1d(input_dimension, tcn_filters, tcn_kernel_size, dilation=1, padding=tcn_kernel_size-1)
        self.bn1 = nn.BatchNorm1d(tcn_filters)
        
        self.conv2 = nn.Conv1d(tcn_filters, tcn_filters, tcn_kernel_size, dilation=2, padding=(tcn_kernel_size-1)*2)
        self.bn2 = nn.BatchNorm1d(tcn_filters)

        self.conv3 = nn.Conv1d(tcn_filters, tcn_filters, tcn_kernel_size, dilation=4, padding=(tcn_kernel_size-1)*4)
        self.bn3 = nn.BatchNorm1d(tcn_filters)
        
        self.conv_res = nn.Conv1d(input_dimension, tcn_filters, kernel_size=1, padding='same') if input_dimension != tcn_filters else None

    def forward(self, x):
        original_x = x.clone()
        if debug_mode_flag: print('Shape of x before conv1 of TCN: ', x.shape)
        x = self.conv1(x)
        x = x[:, :, :-self.conv1.padding[0]]
        if debug_mode_flag: print('Shape of x after conv1 of TCN: ', x.shape)
        x = self.bn1(x)
        x = self.elu(x)
        x = self.dropout(x)

        if debug_mode_flag: print('Shape of x before conv2 of TCN: ', x.shape)
        x = self.conv2(x)
        x = x[:, :, :-self.conv2.padding[0]]
        if debug_mode_flag: print('Shape of x after conv2 of TCN: ', x.shape)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.dropout(x)

        if debug_mode_flag: print('Shape of x before conv3 of TCN: ', x.shape)
        x = self.conv3(x)
        x = x[:, :, :-self.conv3.padding[0]]
        if debug_mode_flag: print('Shape of x after conv3 of TCN: ', x.shape)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.dropout(x)

        if debug_mode_flag: print('Shape of x before conv_res of TCN: ', x.shape)
        res = self.conv_res(original_x) if self.conv_res else original_x
        if debug_mode_flag: print('Shape of res after conv_res of TCN: ', res.shape)
        out = self.elu(x + res)
        
        return out

class ATCNet(nn.Module):
    def __init__(
        self,
        n_classes,
        num_EEG_channels=22,
        # in_samples=1125,
        n_windows=3,
        eegn_F1=16,
        eegn_D=2,
        eegn_kernelSize=64,
        eegn_poolSize=8,
        eegn_dropout=0.3,
        tcn_depth=3,
        tcn_kernelSize=3,
        tcn_filters=32,
        tcn_dropout=0.3,
        tcn_activation='elu',
        fuse='average',
        MSA_embed_dim=512,
        MSA_num_heads=8
    ):
        super(ATCNet, self).__init__()
        self.n_classes = n_classes
        self.num_EEG_channels = num_EEG_channels
        # self.in_samples = in_samples
        self.n_windows = n_windows
        self.eegn_F1 = eegn_F1
        self.eegn_D = eegn_D
        self.eegn_kernelSize = eegn_kernelSize
        self.eegn_poolSize = eegn_poolSize
        self.eegn_dropout = eegn_dropout
        self.tcn_depth = tcn_depth
        self.tcn_kernelSize = tcn_kernelSize
        self.tcn_filters = tcn_filters
        self.tcn_dropout = tcn_dropout
        self.tcn_activation = tcn_activation
        self.fuse = fuse

        self.CNNBlock = CNNBlock(F1=eegn_F1, kernLength=eegn_kernelSize, Chans=num_EEG_channels,
                             D=eegn_D, dropout=eegn_dropout, eegn_poolSize=eegn_poolSize)
        self.dense_layers = nn.ModuleList([
            nn.Linear(eegn_F1 * eegn_D, n_classes) for _ in range(n_windows)
        ])
        self.avg_pool = nn.AvgPool1d(n_windows)
        self.MSA = nn.MultiheadAttention(MSA_embed_dim, MSA_num_heads, dropout=0.5)
        self.tcn = TCNBlock(self.eegn_F1 * self.eegn_D, self.tcn_depth, self.tcn_kernelSize,
                              self.tcn_filters, self.tcn_dropout, self.tcn_activation)

    def forward(self, x):
        # input x shape: (batch_size, num_channels, seq_len) = (batch_size, 22, 1000)
        # print('Shape of x before any execution: ', x.shape)
        x = torch.unsqueeze(x, 1)
        # print('Shape of x before after unsqueeze: ', x.shape)
        # x = x.permute(0, 3, 2, 1)
        # x = x.permute(0, 3, 1, 2)  # similar to Keras Permute layer
        ## expected input shape for CNNBlock is (batch_size, 1, num_channels, seq_len)
        if debug_mode_flag:
            print('Shape of x before CNNBlock: ', x.shape)
        x = self.CNNBlock(x)
        if debug_mode_flag:
            print('Shape of x after CNNBlock: ', x.shape)
        # x = x[:, :, -1, :]
        x = torch.squeeze(x) ## output shape: (batch size, F2, seq_len)
        if debug_mode_flag:
            print('Shape of x before stacked MSA/TCN modules: ', x.shape)

        outputs = []
        ## The sliding window is to seperate the temporal dimension, which is along seq_len
        for i in range(self.n_windows):
            st = i
            end = x.shape[2] - self.n_windows + i + 1
            block2 = x[:, :, st:end] ## (batch size, F2, window_size)

            # if self.attention is not None:
            #     block2 = attention_block(block2, self.attention)
            block2 = block2.permute(2, 0, 1)
            if debug_mode_flag:
                print('Shape of block2 before MSA: ', block2.shape)
            block2 = self.MSA(block2, block2, block2)[0] ## output shape is [seq_len, batch_size, F1*D]
            if debug_mode_flag:
                print('Shape of block2 after MSA: ', block2.shape)
            # x = x.permute(1, 2, 0)

            block3 = self.tcn(block2) ## output shape is [seq_len, F1*D, batch_size]
            if debug_mode_flag:
                print('Shape of block3 after TCNBlock: ', block3.shape)
            ## take the last sequence only
            block3 = block3[-1, :, :] ## output shape is [F1*D, batch_size]
            block3 = block3.permute(1, 0)
            if debug_mode_flag:
                print('Shape of block3 before dense layer: ', block3.shape)

            outputs.append(self.dense_layers[i](block3))
        # print(outputs)

        if self.fuse == 'average':
            if self.n_windows > 1:
                outputs = torch.stack(outputs, dim=0)
                if debug_mode_flag:
                    print('Shape of outputs after stack in average fuse: ', outputs.shape)
                # outputs = outputs.permute(1, 2, 0)
                # outputs = self.avg_pool(outputs)
                outputs = torch.mean(outputs, 0)
                if debug_mode_flag:
                    print('Shape of outputs after average pool in average fuse: ', outputs.shape)
                outputs = outputs.squeeze(dim=0)
            else:
                return outputs[0]
                # outputs = torch.Tensor(outputs[0])
        elif self.fuse == 'concat':
            outputs = torch.cat(outputs, dim=1)
            outputs = self.dense_layers[0](outputs)

        # softmax = F.softmax(outputs, dim=1)
        if debug_mode_flag:
            print('final output shape: ', outputs.shape)

        return outputs #softmax

# %%
binary_classifier_flag = True
# binary_classifier_flag = False
balanced_training_flag = True
debug_mode_flag = False

batch_size = 256  # Set your desired batch size

### Transformer model hyperparameters
segment_interval = 4
resampleFS = 250
sequence_length = resampleFS * segment_interval
num_EEG_channels=22
eegn_F1 = 16
eegn_D = 2
tcn_filters = eegn_F1 * eegn_D

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

MSA_dropout = 0.5  # dropout probability
MSA_embed_dim, MSA_num_heads = eegn_F1*eegn_D, 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = ATCNet( 
            # Dataset parameters
            n_classes = nclasses, 
            num_EEG_channels = num_EEG_channels, 
            # in_samples = sequence_length, 
            # Sliding window (SW) parameter
            n_windows = 5, 
            # Convolutional (CV) block parameters
            eegn_F1 = eegn_F1,
            eegn_D = eegn_D, 
            eegn_kernelSize = 64,
            eegn_poolSize = 7,
            eegn_dropout = 0.3,
            # Temporal convolutional (TC) block parameters
            tcn_depth = 3, 
            tcn_kernelSize = 3,
            tcn_filters = tcn_filters,
            tcn_dropout = 0.3, 
            tcn_activation='elu',
            MSA_embed_dim = MSA_embed_dim,
            MSA_num_heads = MSA_num_heads
            )     

model_stats = summary(model, input_size=(batch_size, num_EEG_channels, sequence_length), verbose=0)
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
else:
    criterion = nn.CrossEntropyLoss()

lr = 0.0001  # learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
STEPLR_period = 20.0
STEPLR_gamma = 0.1
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, STEPLR_period, gamma=STEPLR_gamma)

best_val_loss = float('inf')
epochs = 1000

# Define early stopping parameters
patience = 50
min_delta = 0.01

loss_train_array, loss_val_array = [], []

starting_time = datetime.now()

dt_string = starting_time.strftime("%Y-%m-%d-%H-%M")
print("Starting date and time =", dt_string)

root_path_keywords = 'results_ATCNet_2023_someone_else_model'

if not os.path.exists(root_path_keywords):
    os.mkdir(root_path_keywords)

if binary_classifier_flag:
    results_save_folder = os.path.join(root_path_keywords, 'binary_experiment_' + dt_string)
else:
    results_save_folder = os.path.join(root_path_keywords, 'multi_class_experiment_' + dt_string)
if not os.path.exists(results_save_folder):
    os.mkdir(results_save_folder)

model_save_path = os.path.join(results_save_folder, 'best_model.pt')

# from torchinfo import summary
model_stats = summary(model, input_size=(batch_size, num_EEG_channels, sequence_length), verbose=0)

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

# These variables are for tracking the best score and patience counter
best_model = None
best_score = float('inf')
best_epoch = -1
no_improvement_counter = 0

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
    txtFile.write('Training data root: ' + str(data_root))