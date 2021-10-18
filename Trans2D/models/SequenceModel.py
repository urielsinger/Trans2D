import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import RetrievalNormalizedDCG, RetrievalPrecision
from transformers import BertConfig, BertModel

from .Tokenizer import TensorTokenizer
from .metrics import RetrievalHIT
from .modeling_bert2D import BertModel2D


class SequenceTransformer(nn.Module):
    def __init__(self, hparams):
        """
        Sequence Transformer Model
        Args:
            pooling: str - pooling function on embedding before transformer layer or after (in case of Attention2D)
            transformer_type: str - tyoe of sequence transformer, one of: 'Attention2D', 'Attention', 'GRU', 'LSTM'
            n_attributes: int - number of attributes in each token in the sequence
            hidden_size: int - hidden size to be used by the embedding and transformer layers
            output_size: int - output size for each token
            num_attention_heads: int - number of attention heads, used only in 'Attention2D' and 'Attention'
            num_hidden_layers: int - number of transformer layers
            max_position_embeddings: int - maximum sequence length, used only in 'Attention2D' and 'Attention'
        """
        super(SequenceTransformer, self).__init__()
        assert hparams.transformer_type in ['Attention2D', 'Attention', 'GRU', 'LSTM']
        assert hparams.pooling in ['mean', 'sum', 'transformer', 'concat']

        self.vocab = hparams.vocab
        self.vocab_size = len(self.vocab)
        self.pooling = hparams.pooling
        self.transformer_type = hparams.transformer_type
        self.n_attributes = hparams.n_attributes
        self.hidden_size = hparams.hidden_size
        self.transformer_hidden_size = hparams.hidden_size * self.n_attributes if (self.pooling=='concat' and self.transformer_type!='Attention2D') else hparams.hidden_size
        self.output_hidden_size = hparams.hidden_size * self.n_attributes if self.pooling=='concat' else hparams.hidden_size
        self.output_size = hparams.output_size
        self.num_attention_heads = hparams.num_attention_heads
        self.num_hidden_layers = hparams.num_hidden_layers
        self.max_position_embeddings = hparams.max_position_embeddings

        self.tokenizer = TensorTokenizer(self.vocab.filename, **self.vocab.get_special_tokens())

        padding_idx = self.tokenizer.tokenize(torch.tensor(self.vocab.special_tokens['pad_token']))
        self.embeddings = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=padding_idx)

        config = BertConfig(vocab_size=self.vocab_size,
                            n_attributes=self.n_attributes,
                            pad_token_id=self.vocab.special_tokens['pad_token'],
                            num_attention_heads=self.num_attention_heads,
                            hidden_size=self.transformer_hidden_size,
                            num_hidden_layers=self.num_hidden_layers,
                            max_position_embeddings=self.max_position_embeddings,
                            hidden_dropout_prob=hparams.dropout,
                            attention_probs_dropout_prob=hparams.dropout,
                            )

        self.pre_pooling = self.pooling
        self.post_pooling = None
        if self.transformer_type == 'Attention2D':
            self.transformer = BertModel2D(config, add_pooling_layer=False)
            self.pre_pooling = None
            self.post_pooling = self.pooling
        elif self.transformer_type == 'Attention':
            self.transformer = BertModel(config, add_pooling_layer=False)
        elif self.transformer_type == 'GRU':
            self.transformer = nn.GRU(self.hidden_size, self.hidden_size, num_layers=self.num_hidden_layers)
        elif self.transformer_type == 'LSTM':
            self.transformer = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=self.num_hidden_layers)

        if self.pooling == 'transformer':
            self.feature_attention = nn.Embedding(self.hidden_size, 1)

        self.prediction_layer = nn.Linear(self.output_hidden_size, 1)

    def forward(self, tokens, mask=None):
        '''
        Args:
            tokens: A tensor of size [batch_size X seq_len X n_attributes] representing all attributes
            mask: optional - for padding sequences
        Returns:
            r: A tensor of size [batch_size X seq_len X output_size] with an output for each toekn
        '''
        if mask is None:
            mask = torch.ones_like(tokens[...,0]).bool()

        attributes = self.tokenizer.tokenize(tokens)
        features = self.embeddings(attributes)
        features = self.reduce_tensor(features, pooling=self.pre_pooling)

        if isinstance(self.transformer, BertModel) or isinstance(self.transformer, BertModel2D):
            transformed_features = self.transformer(inputs_embeds=features, attention_mask=mask).last_hidden_state
        elif isinstance(self.transformer, nn.GRU) or isinstance(self.transformer, nn.LSTM):
            transformed_features = self.transformer(features.transpose(0,1))[0].transpose(0,1)

        transformed_features = self.reduce_tensor(transformed_features, pooling=self.post_pooling)

        out = self.prediction_layer(transformed_features)

        return out

    def reduce_tensor(self, tensor, pooling=None):
        if pooling is None:
            return tensor
        elif pooling == 'mean':
            return tensor.mean(dim=-2)
        elif pooling == 'sum':
            return tensor.sum(dim=-2)
        elif pooling == 'transformer':
            attention = torch.softmax(self.feature_attention(tensor), dim=-2)
            return torch.sum(tensor * attention, dim=-2)
        elif pooling == 'concat':
            return tensor.view(*tensor.size()[:-2], -1)



class SequenceTransformerModule(pl.LightningModule):
    def __init__(self, hparams):
        """
        pytorch_lightning module handling the SequenceTransformer model
        """
        super(SequenceTransformerModule, self).__init__()
        self.model = SequenceTransformer(hparams)
        self.criterion = nn.BCEWithLogitsLoss()

        self.K = [1, 2, 5]
        self.learning_rate = hparams.learning_rate
        self.weight_decay = hparams.weight_decay

        metrics = pl.metrics.MetricCollection(dict(**{f'NDCG@{k}': RetrievalNormalizedDCG(k=k) for k in self.K},
                                                   **{f'Precision@{k}': RetrievalPrecision(k=k) for k in self.K},
                                                   **{f'Hit@{k}': RetrievalHIT(k=k) for k in self.K}))
        self.metrics = {'train': metrics.clone(), 'val': metrics.clone(), 'test': metrics.clone()}


    def forward(self, tokens, mask=None):
        return self.model(tokens, mask)

    def step(self, batch, name):
        tokens, labels, prediction_mask = batch['attributes'], batch['labels'], batch['mask']
        indexes = torch.arange(labels.size(0)).unsqueeze(1).repeat(1, labels.size(1))

        logits = self(tokens)

        logits = logits[prediction_mask].squeeze()
        labels = labels[prediction_mask].float().squeeze()
        indexes = indexes[prediction_mask].squeeze()

        loss = self.criterion(logits, labels)
        self.log(f'{name}/loss', loss)

        for metric_name in self.metrics[name]:
            metric = self.metrics[name][metric_name](logits, labels.bool(), indexes)
            self.log(f'{name}/{metric_name}', metric)

        return loss

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        return self.step(batch, name='train')

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        return self.step(batch, name='val')

    def test_step(self, batch: dict, batch_idx: int):
        return self.step(batch, name='test')

    def configure_optimizers(self):
        opt = torch.optim.Adam(params=self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return opt
