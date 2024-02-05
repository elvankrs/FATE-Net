# utils.py

import numpy as np
import random, torch, os
from torch.nn.utils.rnn import pad_sequence

def set_seed(seed: int = 42) -> None:
  np.random.seed(seed)
  random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.use_deterministic_algorithms(True, warn_only=True)
  # When running on the CuDNN backend, two further options must be set
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  # Set a fixed value for the hash seed
  os.environ["PYTHONHASHSEED"] = str(seed)


def convert_to_binary(labels):
  # Convert ordinal labels to binary labels as follows:

  # 2: MCI -> 0
  # 3: AD -> 1
  # 5: MCI to AD -> 1

  seq_lengths = []
  for i in range(len(labels)):
      for j in range(labels[i].shape[0]):
          if (labels[i][j] == 5) or (labels[i][j] == 3) or (labels[i][j] == 6):
              labels[i][j] = 1
          elif (labels[i][j] == 2) or (labels[i][j] == 1) or (labels[i][j] == 4):
              labels[i][j] = 0

  return labels



def anomaly_check_binary(data, labels):
    # Check whether there is any 1-0 pattern.
    # If there is such a pattern, remove the time steps after the transition.

    ctr = 0
    for pat_idx in range(len(labels)):  # Iterate over patients
        converted_to_MCI = False
        for vis_idx in range(labels[pat_idx].shape[0] - 1):  # Iterate over visits of the patient up to the second-to-last visit
            if labels[pat_idx][vis_idx] == 1 and labels[pat_idx][vis_idx + 1] == 0:
                # 1-0 pattern found
                # Discard the visits after turning to MCI
                data[pat_idx] = data[pat_idx][:vis_idx + 1]
                labels[pat_idx] = labels[pat_idx][:vis_idx + 1]
                ctr += 1
                break

    print(f"{ctr} anomalies found.")
    if ctr > 0:
        print(f"WARNING! {ctr} anomalies (1-0 pattern) found.")

    return data, labels


def get_mask_2d(outputs, seq_length):
    # outputs : size (N, T)
    # seq_length : size (T)
    # mask: size (N,T) same shape with outputs, because we want to multiply element-wise
    # For every row of mask matrix,  # the element at (seq_length - 1) is 1, and 0 otherwise.
    if len(outputs.shape) == 2:
        outputs = torch.unsqueeze(outputs, dim=0)
    mask = torch.zeros(outputs.shape[0], outputs.shape[1])
    for i in range(outputs.shape[0]):
        last_idx = seq_length[i] - 1
        mask[i][last_idx] = 1

    return mask



def get_all_steps(labels, seq_lengths):
    # Returns the labels for all visits (while excluding padding)
    list_of_all_steps = []
    for i in range(labels.shape[0]):
        seq_length = int(seq_lengths[i][0])
        steps = (labels[i][:seq_length]).cpu().detach().numpy()
        # steps = steps.astype(np.float)
        list_of_all_steps.append(steps)

    arr_of_all_steps = np.hstack(list_of_all_steps) # first convert to np because elements inside are np arrays
    torch_of_all_steps = torch.from_numpy(arr_of_all_steps)

    return torch_of_all_steps


def get_all_weights(weights, seq_lengths):
    # N T d
    # Returns the weights for all visits (while excluding padding)
    list_of_all_weights = []
    for i in range(weights.shape[0]):
        seq_length = int(seq_lengths[i])
        pat_weights = (weights[i][:seq_length, :]).cpu().detach().numpy()
        list_of_all_weights.append(pat_weights)

    arr_of_all_weights = np.vstack(list_of_all_weights) # first convert to np because elements inside are np arrays
    torch_of_all_weights = torch.from_numpy(arr_of_all_weights)

    return torch_of_all_weights

def accuracy(preds, labels):

  """
  Function to calculate accuracy.

  Parameters:
    preds: predictions -> shape (N, T)
    labels: ground truth labels -> shape (N,T)
    seq_lengths: sequence length for every item in the batch -> shape (N,1)

  Returns:
    accuracy: average accuracy for every step and every item in the batch

  """
  preds = preds.argmax(1)
  accuracy = sum((preds == labels).int()) / labels.shape[0]
  return accuracy


def padding(data, labels=None, max_sequence_length=None):
    """
    Function to apply padding. Returns a tensor with shape (N, T, d) for data
    and a tensor with shape (N, T) for labels.

    Parameters:
    data: list of np.array of shape (T, d)
    labels: list of np.array of shape (T,) or None
    max_sequence_length: int, maximum length for padding

    Returns:
    padded_data: padded torch.tensor of shape (N, T, d)
    padded_labels: padded torch.tensor of shape (N, T) or None
    """

    # Find the dimensionality of each element in the sequence
    dim_data = data[0].shape[1]

    # Create a tensor filled with zeros for data
    padded_data = torch.zeros((len(data), max_sequence_length, dim_data))
    # Copy the data into the tensor
    for i, seq in enumerate(data):
        seq_len = min(len(seq), max_sequence_length)
        seq = np.array(seq, dtype=np.float32)
        padded_data[i, :seq_len, :] = torch.from_numpy(seq[:seq_len]).type(torch.float)

    # If labels are provided, pad them as well
    if labels is not None:
        max_length_labels = max(len(seq) for seq in labels)
        max_length_labels = max_sequence_length 
        padded_labels = torch.zeros((len(labels), max_length_labels), dtype=torch.long)

        for i, seq in enumerate(labels):
            seq_len = min(len(seq), max_length_labels)
            seq = np.array(seq, dtype=int)
            padded_labels[i, :seq_len] = torch.from_numpy(seq[:seq_len])
    else:
        padded_labels = None

    return padded_data, padded_labels



def get_conversion_step(labels, seq_lengths):
  # Returns the patient indices and conversion step indices of AD-converters
  conversion_step_indices, converting_pat_indices = [], []
  for pat_idx in range(labels.shape[0]): # iterate over patients
    MCI_starter = False # default
    if labels[pat_idx][0] == 0:
      MCI_starter = True

    for vis_idx in range(seq_lengths[pat_idx]):
      if (MCI_starter) and (labels[pat_idx][vis_idx] == 1): # AD-converter
        conversion_step_indices.append(vis_idx)
        converting_pat_indices.append(pat_idx)
        break

  converting_pat_indices = np.array(converting_pat_indices)
  conversion_step_indices = np.array(conversion_step_indices)

  return converting_pat_indices, conversion_step_indices



def modify_for_forecasting(data, labels):
  # MODIFYING THE DATASET FOR FORECASTING

  # Cut x from the first step, cut y from the last step
  # X: x1 ... x_t-1
  # Y: y_2 ... y_t

  X, y = [], []

  for i in range(len(data)):
      if (data[i].shape[0] >= 3): # >=3 for FORECASTING
          X.append(data[i][:-1]) # first T-1 time steps FOR FORECASTING
          y.append(labels[i][1:]) # last T-1 time steps FOR FORECASTING
  
  return X, y # Returns the modified versions of data and labels


def get_sequence_lengths(labels):

  seq_lengths = []

  for i in range(len(labels)):
      seq_lengths.append(labels[i].shape[0])

  return np.array(seq_lengths)



def get_batch(x_input, label, time, batch_size):
    num_batch = int(len(x_input) / batch_size)
    inputs = []
    targets = []
    times = []
    start = 0
    for i in range(num_batch-1):
        inputs.append(x_input[start:(start+batch_size)])
        targets.append(label[start:(start+batch_size)])
        times.append(time[start:(start+batch_size)])
        start += batch_size
    inputs.append(x_input[start:])
    targets.append(label[start:])
    times.append(time[start:])
    return inputs, targets, times

def get_time_feature(temporal_data, temporal_labels):
    time = []
    data = []
    labels = []
    for i in range(len(temporal_data)):
        for j in range(len(temporal_data[i])):
            data.append(temporal_data[i][j])
            labels.append(temporal_labels[i][j])
            time.append(j+1)
    return np.array(data, dtype=np.float32), np.array(labels, dtype=int), np.array(time, dtype=int)


def find_max_sequence_length(labels_list):
    max_sequence_length = 0
    for label_array in labels_list:
        sequence_length = label_array.shape[0]  # T: number of time steps
        max_sequence_length = max(max_sequence_length, sequence_length)
    return max_sequence_length

def find_seq_lengths(train_labels, test_labels):
  train_seq_lengths, test_seq_lengths = [], []

  for i in range(len(train_labels)): # FOR TRAINING SET
      train_seq_lengths.append(train_labels[i].shape[0])

  for i in range(len(test_labels)): # FOR TEST SET
      test_seq_lengths.append(test_labels[i].shape[0])

  train_seq_lengths = np.array(train_seq_lengths)
  test_seq_lengths = np.array(test_seq_lengths)

  return train_seq_lengths, test_seq_lengths

def calculate_acc(predicted, labels, seq_len):

  softmax = torch.nn.Softmax(dim=1)
  softmax_predicted = softmax(predicted) # 18,2

  argmax_preds = torch.argmax(softmax_predicted, dim = 1)
  accuracy = sum((argmax_preds[:seq_len] == labels[:seq_len]).int()) / seq_len

  acc_list = (argmax_preds == labels).int()
  acc_list = acc_list[:seq_len]

  return accuracy, acc_list


def accuracy_for_padded(preds, labels, seq_lengths):
  """
  Function to calculate accuracy.

  Parameters:
    preds: predictions -> shape (N, T)
    labels: ground truth labels -> shape (N,T)
    seq_lengths: sequence length for every item in the batch -> shape (N,1)

  Returns:
    accuracy: average accuracy for every step and every item in the batch

  """
  if len(preds.shape)==2:
      preds = preds.unsqueeze(dim=0)
  softmax = torch.nn.Softmax(dim=2)
  softmax_predicted = softmax(preds) # 18,2
  preds = torch.argmax(softmax_predicted, dim = 2)
  preds_all_steps = get_all_steps(preds, seq_lengths)
  labels_all_steps = get_all_steps(labels, seq_lengths)
  accuracy = sum((preds_all_steps == labels_all_steps).int()) / labels_all_steps.shape[0]
  return accuracy
