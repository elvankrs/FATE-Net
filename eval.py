import numpy as np
import pandas as pd
import argparse, json, torch
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, roc_auc_score
import utils
from model import AttentionWithTimeEmbedding, LSTMModelWithTimeEmbedding, CNN1DModelWithTimeEmbedding, TransformerEncoderWithTimeEmbedding

torch.set_printoptions(threshold=1000000000)
np.set_printoptions(threshold=1000000000)
pd.options.display.float_format = '${:,.3f}'.format


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
final_dim = config['final_dim']
proj_dim = config['proj_dim']
model_type = config['model_type']
model_path = config['model_dir'] + '/' + model_type + 'checkpoint-'+ str(max_epoch) +'.pth'
dataset = config['dataset']

# Load the test set
test_data = np.load(config['data_path'] + "test_next_data.npy", allow_pickle=True) 
test_labels = np.load(config['data_path'] + "test_next_label.npy", allow_pickle=True)
utils.set_seed(42)

# Modify the dataset for forecasting # hocanın datası forecasting için olduğundan bu adımı atlıyorum
test_data, test_labels = np.array(test_data), np.array(test_labels)
# Convert multiclass labels to binary labels
if dataset=="ADNI":
    test_labels = utils.convert_to_binary(test_labels)
# Prints a warning message if there is an anomaly
test_data, test_labels = utils.anomaly_check_binary(test_data, test_labels)
# Get the sequence lengths
test_seq_lengths = utils.get_sequence_lengths(test_labels)
# Find the maximum sequence length
max_sequence_length = utils.find_max_sequence_length(test_labels)
# Apply padding
test_data_padded, test_label_padded = utils.padding(test_data, test_labels, max_sequence_length=max_sequence_length)
test_labels_yedek = test_labels

# Convert temporal data to cross-sectional data
test_data, test_labels, test_time = utils.get_time_feature(test_data, test_labels)

COG_FEATURES = 8 # number of cognitive score features
MRI_FEATURES = 6

# Get features
test_cog_sequences = test_data[:, :COG_FEATURES]
test_mri_sequences = test_data[:, COG_FEATURES:]


# Load the trained model
if model_type=="time_emb":
    model = AttentionWithTimeEmbedding(embed_dim, proj_dim, final_dim)
elif model_type=="LSTM_time_emb":
    model = LSTMModelWithTimeEmbedding(input_dim=64, hidden_dim=2*final_dim, embed_dim=embed_dim, proj_dim=proj_dim, final_dim=final_dim)
elif model_type=="1DCNN_time_emb":
    model = CNN1DModelWithTimeEmbedding(input_dim=proj_dim, hidden_dim=final_dim, embed_dim=embed_dim, proj_dim=proj_dim, final_dim=final_dim)
elif model_type=="transformer_time_emb":
    model = TransformerEncoderWithTimeEmbedding(input_dim=proj_dim, embed_dim=embed_dim, proj_dim=proj_dim, final_dim=final_dim)

checkpoint = torch.load(model_path,map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model'])


model.eval()

test_cog_sequences, test_mri_sequences, test_time = torch.tensor(test_cog_sequences), torch.tensor(test_mri_sequences), torch.tensor(test_time, dtype=torch.float32)
test_sequences = torch.concatenate((test_cog_sequences, test_mri_sequences), dim=1)

if model_type=="time_emb":
    preds, attention_weights = model(test_sequences, test_time.unsqueeze(1))
    attention_weights = torch.squeeze(attention_weights)

else:
    preds, attention_weights = model(test_data_padded) 
    attention_weights = attention_weights.permute(1,2,0)
    all_weights = utils.get_all_weights(attention_weights, test_seq_lengths)

preds = torch.squeeze(preds)
print("\nTesting time")

test_seq_lengths= np.reshape(test_seq_lengths, (-1,1))

if model_type != "time_emb":
    # All steps
    if model_type=="LSTM" or model_type=="1DCNN" or model_type=="transformer" or model_type=="LSTM_time_emb" or model_type=="1DCNN_time_emb" or model_type=="transformer_time_emb":
        preds = preds.argmax(2)
        preds_all_steps = utils.get_all_steps(preds, test_seq_lengths)
        labels_all_steps = utils.get_all_steps(test_label_padded, test_seq_lengths)

if model_type=="time_emb":
    preds = preds.argmax(1)

print("Classification report for ALL steps")
target_names = ['MCI', 'AD']

if model_type == "LSTM" or model_type=="1DCNN" or model_type=="transformer" or model_type=="LSTM_time_emb" or model_type=="1DCNN_time_emb" or model_type=="transformer_time_emb":
    test_labels = labels_all_steps
    preds = preds_all_steps


print(classification_report(test_labels, preds, target_names=target_names))
# Gives UndefinedMetricWarning since there is no predicted samples for one class.

MCI_prec, AD_prec = precision_score(test_labels, preds, average=None)
MCI_recall, AD_recall = recall_score(test_labels, preds, average=None)
MCI_F1, AD_F1 = f1_score(test_labels, preds, average=None)

avg_prec = precision_score(test_labels, preds, average = 'weighted')
avg_recall = recall_score(test_labels, preds, average = 'weighted')
avg_F1 = f1_score(test_labels, preds, average = 'weighted')
roc_auc = roc_auc_score(test_labels, preds, average='weighted')

print(f"\nWeighted average of precision_score: {avg_prec:.3f}")
print(f"Weighted average of recall_score: {avg_recall:.3f}")
print(f"Weighted average of f1_score: {avg_F1:.3f}")
print(f"ROC-AUC: {roc_auc:.3f}\n")


# Conversion steps
print("Classification report for CONVERSION steps")
test_seq_lengths = test_seq_lengths.squeeze()
converting_pat_indices, conversion_step_indices = utils.get_conversion_step(test_label_padded, test_seq_lengths) #test_labels


conv_cog_sequences, conv_mri_sequences, conv_times, conv_labels = [], [], [], []
for i in range(converting_pat_indices.shape[0]):
  pat_idx = converting_pat_indices[i]
  vis_idx = conversion_step_indices[i]
  conv_cog_sequences.append(np.array(test_data_padded[pat_idx][vis_idx][:COG_FEATURES]))
  conv_mri_sequences.append(np.array(test_data_padded[pat_idx][vis_idx][COG_FEATURES:]))
  conv_times.append(vis_idx + 1)
  conv_labels.append(test_label_padded[pat_idx][vis_idx])


conv_cog_sequences = torch.tensor(np.array(conv_cog_sequences))
conv_mri_sequences = torch.tensor(np.array(conv_mri_sequences))
conv_time = torch.tensor(np.array(conv_times), dtype=torch.float32)

conv_data = torch.concatenate((conv_cog_sequences, conv_mri_sequences), axis=-1)
conv_data = conv_data.unsqueeze(1)

if model_type=="LSTM_time_emb" or model_type=="1DCNN_time_emb" or model_type=="transformer_time_emb":
    conv_data_padded, _ = utils.padding(conv_data, max_sequence_length=max_sequence_length)
    conv_preds, conv_attention_weights = model(conv_data_padded)
    conv_preds = conv_preds.argmax(2)
    conv_tp_indices = np.where(conv_preds == 1)[0] # conversion true positive indices
    conv_preds = conv_preds[:, 0]


elif model_type=="time_emb":

    conv_sequences = torch.concatenate((conv_cog_sequences, conv_mri_sequences), axis=-1)

    conv_preds, conv_attention_weights = model(conv_sequences, conv_time.unsqueeze(1))
    conv_preds = torch.squeeze(conv_preds)
    conv_preds = conv_preds.argmax(1)
    conv_attention_weights = torch.squeeze(conv_attention_weights)

    conv_tp_indices = torch.where(conv_preds == 1)[0] # conversion true positive indices


print(classification_report(conv_labels, conv_preds, target_names=target_names))
# Gives "ValueError: Number of classes, 1, does not match size of target_names, 2. Try specifying the labels parameter"
# when all predicted labels are MCI but they are all AD. (It works fine when network predicts AD for at least one patient.)

conv_prec = precision_score(conv_labels, conv_preds)
conv_recall = recall_score(conv_labels, conv_preds)
conv_f1 = f1_score(conv_labels, conv_preds)

print(f"precision_score: {precision_score(conv_labels, conv_preds):.3f}")
print(f"recall_score: {recall_score(conv_labels, conv_preds):.3f}")
print(f"f1_score: {f1_score(conv_labels, conv_preds):.3f}")



