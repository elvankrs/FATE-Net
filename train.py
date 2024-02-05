import os, argparse, json, torch, pickle
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score


import utils
from prep import prep
from model import AttentionWithTimeEmbedding, LSTMModelWithTimeEmbedding, CNN1DModelWithTimeEmbedding, TransformerEncoderWithTimeEmbedding


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str)
args = parser.parse_args()

with open(args.config+'.json') as config_file:
    config = json.load(config_file)


learning_rate = config['learning_rate']
batch_size = config['batch_size']
max_epoch = config['max_epoch']
model_dir = config['model_dir']
embed_dim = config['embed_dim']
proj_dim = config['proj_dim']
final_dim = config['final_dim']
model_type = config['model_type']
dataset = config['dataset']
data_path = config['data_path']

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))


# Load the train and test sets
train_data = np.load(config['data_path'] + "training_next_data.npy", allow_pickle=True) 
train_labels = np.load(config['data_path'] + "training_next_label.npy", allow_pickle=True)
test_data = np.load(config['data_path'] + "test_next_data.npy", allow_pickle=True) 
test_labels = np.load(config['data_path'] + "test_next_label.npy", allow_pickle=True)


utils.set_seed(42)

attributes = ['CDRSB', 'ADAS11', 'ADAS13', 'MMSE', 'RAVLT_immediate', 'RAVLT_learning', 'RAVLT_forgetting', 'FAQ', 'Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp']

cross_sectional_models = ["time_emb", "self_attn"]
temporal_models = ["LSTM_time_emb", "1DCNN_time_emb", "transformer_time_emb"]

prep_mode = "cross_sec" if model_type in cross_sectional_models else "return_padded"

data = prep(train_data, train_labels, test_data, test_labels, batch_size=batch_size, mode=prep_mode, dataset=dataset)
train_loader, val_loader = data["train_loader"], data["val_loader"]

if model_type=="time_emb":
  model = AttentionWithTimeEmbedding(embed_dim=embed_dim, proj_dim=proj_dim, final_dim=final_dim).to(device)
elif model_type=="LSTM_time_emb":
  model = LSTMModelWithTimeEmbedding(input_dim=proj_dim, hidden_dim=2*final_dim, embed_dim=embed_dim, proj_dim=proj_dim, final_dim=final_dim).to(device)
elif model_type=="1DCNN_time_emb":
  model = CNN1DModelWithTimeEmbedding(input_dim=proj_dim, hidden_dim=final_dim, embed_dim=embed_dim, proj_dim=proj_dim, final_dim=final_dim).to(device)
elif model_type=="transformer_time_emb":
  model = TransformerEncoderWithTimeEmbedding(input_dim=proj_dim, embed_dim=embed_dim, proj_dim=proj_dim, final_dim=final_dim).to(device)

model.train()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

losses, train_accuracies, val_accuracies = [], [], []
itr = 0

for epoch in range(max_epoch):
  
  model.train()
  epoch_loss, epoch_train_acc = 0, 0

  for batch_idx, (batch) in enumerate(train_loader):

    sequences = ((batch['sequence']).float()).requires_grad_().to(device)
    labels = (batch['label']).long().to(device) # shape (batch_size, MAX_LENGTH)
    times = ((batch['time']).float()).to(device)

    if model_type not in cross_sectional_models:
      seq_lengths = ((batch['seq_length']).int()).to(device)

    if model_type=="time_emb":
      outputs, attention_weights = model(sequences, times.unsqueeze(1))
    else:
      outputs, attention_weights = model(sequences)

    outputs = torch.squeeze(outputs)

    if model_type not in cross_sectional_models:
      seq_lengths = torch.reshape(seq_lengths, (-1, 1))
    if model_type in cross_sectional_models:
      train_acc = utils.accuracy(outputs, labels)
    else:
      train_acc = utils.accuracy_for_padded(outputs, labels, seq_lengths)


    epoch_train_acc += train_acc
    if model_type in temporal_models: # they require masking since they take padded data
      mask =  utils.get_mask_2d(outputs, seq_lengths).to(device)
      outputs = torch.mul(mask[:, :, None], outputs)

    criterion = nn.CrossEntropyLoss() # Loss function with modified weights

    if model_type not in cross_sectional_models:
      outputs = torch.transpose(outputs, 1, 2)
    loss = criterion(outputs, labels) # binary cross entropy loss

    epoch_loss += loss

    optimizer.zero_grad() # clear the previous gradients
    loss.backward()
    optimizer.step()

    itr += batch['label'].size(0)

    if itr % batch_size == 0:

      model.eval()
      val_total_acc, pat_ctr = 0, 0

      for data in val_loader:

        model.eval()

        val_inputs = (data['sequence']).float().to(device)
        val_times = ((data['time']).float()).requires_grad_().to(device)
        val_labels = data['label'].to(device)
        if model_type not in cross_sectional_models:
          val_seq_lengths = ((data['seq_length']).int()).unsqueeze(dim=-1).to(device)

        if model_type=="time_emb":
          val_outputs, _ = model(val_inputs, val_times.unsqueeze(1)) # forward pass to validation data
        else:
          val_outputs, _ = model(val_inputs)

        predicted = torch.squeeze(val_outputs, dim=-1)

        if model_type in cross_sectional_models:
          pat_acc = utils.accuracy(predicted, val_labels)
        else:        
            pat_acc = utils.accuracy_for_padded(predicted.cpu(), val_labels.cpu(), val_seq_lengths.cpu())
        pat_ctr += 1
        val_total_acc += pat_acc # Add patient's accuracy to batch accuracy


  epoch_avg_loss = epoch_loss / (batch_idx + 1)
  epoch_train_avg_acc = epoch_train_acc / (batch_idx + 1)
  val_acc = val_total_acc / pat_ctr

  losses.append(epoch_avg_loss.cpu().detach().numpy())
  train_accuracies.append(epoch_train_avg_acc)
  val_accuracies.append(val_acc)

  if epoch % 1 == 0 or (epoch+1) == max_epoch:
    print('Epoch: {}.\tIteration: {}.\tLoss: {:.5}\tTrain accuracy: {:.3f}\tVal accuracy: {:.3f}'.format(epoch, itr, epoch_avg_loss, epoch_train_avg_acc, val_acc))
  
  if (epoch+1) % 100 == 0 or (epoch+1) == max_epoch:
    file_path = os.path.join(model_dir, (model_type + 'checkpoint-%s.pth' % (epoch+1)))
    state = {
              'model': model.state_dict(),
              'epoch': epoch,
              'optimizer': optimizer.state_dict()
              }
    if not os.path.isdir(model_dir):
          os.mkdir(model_dir)

    torch.save(state, file_path)

print(f"Training is finished!\n")


# Plotting the results

fig = plt.figure()
fig.set_size_inches(12, 6)
loss_plot = fig.add_subplot(221)
acc_plot = fig.add_subplot(223)
loss_plot.title.set_text(f"Loss (lr = {learning_rate}, {max_epoch} epochs)")
acc_plot.title.set_text("Accuracy")
loss_plot.set_xlabel("Number of epochs")
loss_plot.set_ylabel("Loss")
acc_plot.set_xlabel("Number of epochs")
acc_plot.set_ylabel("Accuracy")
loss_plot.plot(losses)
acc_plot.plot(train_accuracies)
acc_plot.plot(val_accuracies)

plt.legend(["train", "val"], loc ="lower right")
plt.subplots_adjust(left=0.1,bottom=0.1,right=0.9,top=0.9,wspace=0.4,hspace=0.4)
plt.show()
fig_name = "lr_" + str(learning_rate)[2:] + "_epoch_" + str(max_epoch) + "_" + data_path[-4:-1]
plt.savefig(fname=fig_name)
