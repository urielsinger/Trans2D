from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import pytorch_lightning as pl

from .vocab import Vocabulary


class SequenceDataModule(pl.LightningDataModule):
    """
    Example of a DataModule handling data for the SequenceTransformer model.
    Currently only generated random data.
    """
    def __init__(self ,dataset_name, max_seq_len, batch_size=32, num_workers=0):
        super().__init__()
        self.dataset_name = dataset_name
        self.max_seq_len = max_seq_len

        self.special_tokens = OrderedDict([('pad_token',  0),
                                           ('unk_token', -1),
                                           ('mask_token', -2)])

        self.n_users = None
        self.n_attributes = None

        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        """
        Prepares the dataset
        Returns:
            user2seq - dict holding for each user a dictionary holding:
                            - attributes: list of attributes for each interaction
                            - timestamps: list of timestamp for each interaction
                            - labels: 1 or 0 for each interaction
        """
        self.user2seq = {}
        if self.dataset_name == 'random':
            self.n_users = 100
            self.n_attributes = 2
            self.attribure_max_vocab_size = 10

            # generate data
            for user in torch.range(1, self.n_users, dtype=torch.long):
                seq_len = torch.randint(low=0, high=self.max_seq_len, size=(1,))[0]
                self.user2seq[user] = {'attributes': torch.randint(low=0, high=self.attribure_max_vocab_size, size=(seq_len, self.n_attributes)),
                                       'timestamps': torch.sort(torch.randint(low=0, high=self.max_seq_len*self.n_users, size=(seq_len,)))[0],
                                       'labels': torch.randint(low=0, high=2, size=(seq_len,))}
        else:
            raise Exception(f'dataset {self.dataset_name} not supported')

        # reindex attributes not to overlap and to start from 1
        attributes = torch.cat([seq['attributes'] for seq in self.user2seq.values()], dim=0)
        attribure_min_index = torch.min(attributes, dim=0)[0]
        attribure_max_index = torch.max(attributes, dim=0)[0]
        attribure_vocab_size = (attribure_max_index - attribure_min_index + 1).cumsum(dim=-1)
        attribure_vocab_size = torch.cat([torch.zeros(1).long(), attribure_vocab_size[:-1]]).unsqueeze(0)
        attribure_min_index = attribure_min_index.unsqueeze(0)
        for seq in self.user2seq.values():
            seq['attributes'] = seq['attributes']  - attribure_min_index + 1 + attribure_vocab_size

        # save vocabulary
        self.vocab = Vocabulary(self.special_tokens)
        tokens = torch.unique(torch.cat([seq['attributes'] for seq in self.user2seq.values()], dim=0))
        self.vocab.add_tokens(tokens.tolist())
        self.vocab.save_vocab(f"{self.dataset_name}_vocab.nb")

        assert not (tokens[..., None] == torch.Tensor(list(self.special_tokens.values()))).any()

    def setup(self, stage=None):
        """
        Splits the data into train/val/test by splitting the timeline
        """
        self.datasets = []
        all_timestamps = torch.cat([seq['timestamps'] for seq in self.user2seq.values()])
        qs = (0., 0.7, 0.85, 1.)
        for i in range(1, len(qs)): # split to train/val/test on timeline
            q = torch.Tensor([qs[i - 1], qs[i]])
            start_timestamp, end_timestamp = torch.quantile(all_timestamps.float(), q=q).tolist()

            cur_user2seq = {}
            for user in self.user2seq:
                seq = self.user2seq[user]
                future_mask = seq['timestamps'] <= end_timestamp
                time_mask = (start_timestamp <= seq['timestamps']) * (seq['timestamps'] <= end_timestamp)
                if time_mask.sum() > 0:
                    cur_user2seq[user] = {'attributes': seq['attributes'][future_mask],
                                          'timestamps': seq['timestamps'][future_mask],
                                          'labels': seq['labels'][future_mask],
                                          'mask': time_mask[future_mask]}
            self.datasets.append(SequenceDataset(cur_user2seq, max_seq_len=self.max_seq_len ,special_tokens=self.special_tokens))

    def train_dataloader(self):
        return DataLoader(self.datasets[0], shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.pad_collate)

    def val_dataloader(self):
        return DataLoader(self.datasets[1], shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.pad_collate)

    def test_dataloader(self):
        return DataLoader(self.datasets[2], shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.pad_collate)

    def pad_collate(self, batch):
        """
        This function is responsible of merging all samples into a full batch.
        """
        if isinstance(batch[0], dict):
            return {k: self.pad_collate([sample[k] for sample in batch]) for k in batch[0]}

        max_len = max([len(sample) for sample in batch])
        base_pad = (0, 0) * (batch[0].dim() - 1)
        for i in range(len(batch)):
            cur_len = len(batch[i])
            pad_len = max_len - cur_len
            batch[i] = F.pad(input=batch[i], pad=(*base_pad, pad_len, 0), mode='constant', value=self.special_tokens['pad_token'])
        return default_collate(batch)


class SequenceDataset(Dataset):
    def __init__(self, user2seq, max_seq_len, special_tokens):
        super(SequenceDataset, self).__init__()
        self.user2seq = user2seq
        self.special_tokens = special_tokens
        self.users = list(self.user2seq.keys())

        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        """
        Returns:
            return one sample.
        """
        user = self.users[index]
        seq = self.user2seq[user]

        attributes = seq['attributes']
        labels = seq['labels']
        mask = seq['mask']

        return {'attributes': attributes[-self.max_seq_len:], 'labels': labels[-self.max_seq_len:], 'mask': mask[-self.max_seq_len:]}