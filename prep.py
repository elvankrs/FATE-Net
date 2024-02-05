import utils

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader

# IMPORTANT: These parameters may change if the dataset version changes. The current version is from 04.01.23.
# MAX_LENGTH = 18
COG_FEATURES = 8
MRI_FEATURES = 6

# Preparing dataset and data loader

class TadpoleDataset(Dataset):

    def __init__(self, sequences, labels, times, seq_lengths):
        self.sequences = sequences
        self.labels = labels
        self.times = times
        self.seq_lengths = seq_lengths

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        time = self.times[idx]
        seq_length = self.seq_lengths[idx] # for temporal models

        return dict(
            sequence = torch.Tensor(sequence),
            label = torch.tensor(label).clone().detach().float(),
            # label = torch.tensor(label),
            time = torch.tensor(time).clone().detach().float(),
            seq_length = torch.tensor(seq_length).clone().detach().int() # for temporal models
        )


class TadpoleDataModule(pl.LightningDataModule):

    def __init__(self, train_sequences, train_labels, train_time, train_seq_lengths,
                 test_sequences, test_labels, test_time, test_seq_lengths, batch_size, val_batch_size):
        super().__init__()
        self.train_sequences = train_sequences
        self.train_labels = train_labels
        self.train_time = train_time
        self.train_seq_lengths = train_seq_lengths
        self.test_sequences = test_sequences
        self.test_labels = test_labels
        self.test_time = test_time
        self.test_seq_lengts = test_seq_lengths
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size

    def setup(self):
        self.train_dataset = TadpoleDataset(self.train_sequences, self.train_labels, self.train_time, self.train_seq_lengths)
        self.test_dataset = TadpoleDataset(self.test_sequences, self.test_labels, self.test_time, self.test_seq_lengts)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            shuffle = False
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size = self.val_batch_size,
            shuffle = False
        )
    
  

def prep(train_data, train_labels, test_data, test_labels, batch_size, mode, dataset):
  
  # Takes raw data and outputs the final form to be fed into the model
  # mode: Can be "cross_sec" - "return_padded"
  # data: Can be "ADNI" - "NACC"

  train_data, train_labels, test_data, test_labels = np.array(train_data), np.array(train_labels), np.array(test_data), np.array(test_labels)

  # Convert multiclass labels to binary labels (for ADNI)
  if dataset=="ADNI":
    train_labels = utils.convert_to_binary(train_labels)
    test_labels = utils.convert_to_binary(test_labels)

  # Prints a warning message if there is an anomaly
  train_data, train_labels = utils.anomaly_check_binary(train_data, train_labels)
  test_data, test_labels = utils.anomaly_check_binary(test_data, test_labels)

  # Get the sequence lengths
  train_seq_lengths = utils.get_sequence_lengths(train_labels)
  test_seq_lengths = utils.get_sequence_lengths(test_labels)
  
  #assert MAX_LENGTH == np.max(train_seq_lengths), f"WARNING! MAX_LENGTH={MAX_LENGTH} is not equal to maximum sequence length={np.max(train_seq_lengths)}."

  # Find the maximum sequence length
  max_sequence_length = utils.find_max_sequence_length(np.concatenate((train_labels, test_labels)))

  # Apply padding
  if mode=="return_padded": # for temporal models (LSTM, 1DCNN and transformer)
    train_data_padded, train_label_padded = utils.padding(train_data, train_labels, max_sequence_length=max_sequence_length)
    test_data_padded, test_label_padded = utils.padding(test_data, test_labels, max_sequence_length=max_sequence_length)
  
  # Get time feature
  train_data, train_labels, train_time = utils.get_time_feature(train_data, train_labels)
  test_data, test_labels, test_time = utils.get_time_feature(test_data, test_labels)

  # Form the pytorch dataset

  TEST_BATCH_SIZE = test_data.shape[0]

  if mode=="cross_sec":
     data_module = TadpoleDataModule(train_data, train_labels, train_time, train_seq_lengths,
                                  test_data, test_labels, test_time, test_seq_lengths, batch_size = batch_size, val_batch_size=TEST_BATCH_SIZE)
  elif mode=="return_padded":
     data_module = TadpoleDataModule(train_data_padded, train_label_padded, train_time, train_seq_lengths,
                                  test_data_padded, test_label_padded, test_time, test_seq_lengths, batch_size = batch_size, val_batch_size=TEST_BATCH_SIZE)
   
  
  data_module.setup()

  # Data loaders
  training_loader = data_module.train_dataloader()
  validation_loader = data_module.val_dataloader()

  return {
      'train_loader': training_loader, 'val_loader': validation_loader,
       'batch_size': batch_size,
       'test_data': test_data,
       'test_labels': test_labels, 'test_time': test_time,
       'test_data_padded': test_data, 'test_label': test_labels, 'test_seq_lengths': test_seq_lengths
       }



