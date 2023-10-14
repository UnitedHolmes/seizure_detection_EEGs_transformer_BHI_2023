'''
Adapted from EEG-TCNet: An Accurate Temporal Convolutional Network for Embedded Motor-Imagery Brain–Machine Interfaces
Adapted from Electroencephalography-based motor imagery classification using temporal convolutional network fusion

'''


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



class EEGNet(nn.Module):
    def __init__(self, sequence_length=1000, nb_classes=2, F1=16, eegnet_kernel_size=32, D=2, eeg_chans=22, eegnet_separable_kernel_size=16,
                 eegnet_pooling_1=4, eegnet_pooling_2=8, dropout=0.5):
        super(EEGNet, self).__init__()

        # self.F1 = F1
        # self.eegnet_kernel_size = eegnet_kernel_size
        # self.D = D
        F2 = F1*D
        self.dropout = nn.Dropout(dropout)

        self.block1 = nn.Conv2d(1, F1, (1, eegnet_kernel_size), padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        # self.block2 = nn.Conv2d(F1, F1*D, (eeg_chans, 1), groups=F1, padding='valid', bias=False)
        self.block2 = nn.Conv2d(F1, F1*D, (eeg_chans, 1), padding='valid', bias=False)
        self.bn2 = nn.BatchNorm2d(F1*D)
        self.elu = nn.ELU()
        self.avg_pool1 = nn.AvgPool2d((1, eegnet_pooling_1))
        # self.block3 = nn.Conv2d(F1*D, F2, (1, eegnet_separable_kernel_size), padding='same', groups=F1*D, bias=False)
        self.block3 = nn.Conv2d(F1*D, F2, (1, eegnet_separable_kernel_size), padding='same', bias=False)
        self.bn3 = nn.BatchNorm2d(F1*D)
        self.avg_pool2 = nn.AvgPool2d((1, eegnet_pooling_2))
        self.linear = nn.Linear(F1*D*(sequence_length//eegnet_pooling_1//eegnet_pooling_2), nb_classes)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        # x.shape = (B, 1, C, L)

        x = self.block1(x)
        # x.shape = (B, F1, C, L)
        if debug_mode_flag: print('Shape of x after block1 of EEGNet: ', x.shape)

        x = self.bn1(x)
        x = self.block2(x)
        # x.shape = (B, F1*D, 1, L)
        if debug_mode_flag: print('Shape of x after block2 of EEGNet: ', x.shape)

        x = self.bn2(x)
        x = self.elu(x)
        x = self.avg_pool1(x)
        x = self.dropout(x)
        # x.shape = (B, F1*D, 1, L/8)
        if debug_mode_flag: print('Shape of x before block3 of EEGNet: ', x.shape)
        x = self.block3(x)
        # x.shape = (B, F1*D, 1, L/8)
        if debug_mode_flag: print('Shape of x after block3 of EEGNet: ', x.shape)

        x = self.bn3(x)
        x = self.elu(x)
        x = self.avg_pool2(x)
        x = self.dropout(x)
        # x.shape = (B, F1*D, 1, L/64)
        x = torch.squeeze(x)
        x = x.reshape(x.shape[0], -1)

        if debug_mode_flag: print('Shape of x before linear layer: ', x.shape)
        x = self.linear(x)

        if debug_mode_flag: print('Shape of x by the end of EEGNet: ', x.shape)

        return x


binary_classifier_flag = True
# binary_classifier_flag = False
balanced_training_flag = True
debug_mode_flag = False

batch_size = 2048  # Set your desired batch size

### model hyperparameters
eegnet_F1, eegnet_D = 16, 2
eegnet_kernel_size = 64

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

dropout = 0.3  # dropout probability

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = EEGNet(nb_classes = nclasses, eeg_chans=num_eeg_channels, sequence_length=sequence_length,
                 F1=eegnet_F1, D=eegnet_D, eegnet_kernel_size=eegnet_kernel_size).to(device)

model_stats = summary(model, input_size=(batch_size, num_eeg_channels, sequence_length), verbose=0)
print(model_stats)


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
STEPLR_period = 10.0
STEPLR_gamma = 0.1
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, STEPLR_period, gamma=STEPLR_gamma)

best_val_loss = float('inf')
epochs = 1000
best_model = None

loss_train_array, loss_val_array = [], []

starting_time = datetime.now()

dt_string = starting_time.strftime("%Y-%m-%d-%H-%M")
print("Starting date and time =", dt_string)

root_path_keywords = 'results_EEGNet_only_someone_else_model'

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
best_model = None
best_score = float('inf')
best_epoch = -1
no_improvement_counter = 0
test_macro_f1_score = []

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


