from multiprocessing.sharedctypes import Value
import torch
import torch.nn as nn
from torch.nn import functional
from transformers import AutoModel, AutoConfig, BertConfig
from transformers.models.bert import BertModel


# !note leave to modify
class InvRNN(nn.Module):
    """
    A RNN-based invariant rationalization model.
    Args:

    """
    def __init__(
        self, 
        emb_args,
        hidden_size,
        num_class,
        num_env
    ):
        """
        Inputs:
            emb_args: a dict storing embedding arguments
                type: word2vec or transformer model name
                num_emb
                emb_dim
                pretrain_emb
        """
        super().__init__()

        self.is_transformer = not(emb_args['type'] == 'word2vec')

        # initialize embedding layers
        if self.is_transformer:
            self.gen_embed_layer = AutoModel.from_pretrained(emb_args['type'])
            self.env_inv_embed_layer = AutoModel.from_pretrained(emb_args['type'])
            self.env_enable_embed_layer = AutoModel.from_pretrained(emb_args['type'])
            emb_dim = self.gen_embed_layer.config.hidden_size
        else:
            self.gen_embed_layer = nn.Embedding(emb_args['num_emb'], emb_args['emb_dim'])
            self.env_inv_embed_layer = nn.Embedding(emb_args['num_emb'], emb_args['emb_dim'])
            self.env_enable_embed_layer = nn.Embedding(emb_args['num_emb'], emb_args['emb_dim'])
            if 'pretrained_emb' in emb_args:
                print('Load pretrained embeddings.')
                self.gen_embed_layer.from_pretrained(torch.Tensor(emb_args['pretrained_emb']))
                self.env_inv_embed_layer.from_pretrained(torch.Tensor(emb_args['pretrained_emb']))
                self.env_enable_embed_layer.from_pretrained(torch.Tensor(emb_args['pretrained_emb']))
            emb_dim = emb_args['emb_dim']
        # initialize rational generator
        self.generator = nn.GRU(
            emb_dim,
            hidden_size,
            num_layers = 1,
            bidirectional = True,
            batch_first = True
            )
        
        self.generator_fc = nn.Linear(hidden_size * 2, 2) # shape is 2 as mask or not mask

        # initialize RNN encoders for the predictors
        self.env_inv_encoder = nn.GRU(
            emb_dim,
            hidden_size,
            num_layers = 1,
            bidirectional = True,
            batch_first = True
        )
        

        self.env_enable_encoder = nn.GRU(
            emb_dim + num_env,
            hidden_size,
            num_layers = 1,
            bidirectional = True,
            batch_first = True
        )
        

        # initialize output layer (classification task)
        self.env_inv_fc = nn.Linear(hidden_size * 2, num_class)
        self.env_enable_fc = nn.Linear(hidden_size * 2, num_class)
    
    def straight_through_sampling(self, logits):
        """
        Input:
            logits -- (batch, seq_len, )
        """
        z = functional.softmax(logits, dim = -1)
        z_hard = functional.one_hot(torch.argmax(z, dim = -1), num_classes = z.shape[-1])
        # z_hard.requires_grads is False
        new_hard = z_hard - z.data + z

        return new_hard
        
    def forward(
        self,
        text_ids,
        mask,
        env
    ):
        """
        Inputs:
            text_ids -- (batch_size, seq_len)
            mask -- (batch_size, seq_len)
            env -- (batch_size, num_envs)
        Outputs:
            rationale -- (batch_size, seq_len, 2)
            env_inv_logits -- (batch_size, num_class)
            env_enable_logits -- (batch_size, num_class)
        """

        # aviod warning of RNN weights are not contiguous
        self.generator.flatten_parameters()
        self.env_inv_encoder.flatten_parameters()
        self.env_enable_encoder.flatten_parameters()

        # expand mask
        mask_ = mask.unsqueeze(dim = -1)
        device = text_ids.device
        all_ones = torch.ones(text_ids.shape).unsqueeze(dim = -1).to(device)
        all_zeros = torch.zeros(all_ones.shape).to(device)

        # ########## generator ##########
        gen_embeddings = mask_ * self.gen_embed_layer(text_ids)
        gen_outputs, _ = self.generator(gen_embeddings)
        gen_logits = self.generator_fc(gen_outputs)

        # generate rationale (batch_size, seq_len, 2)
        # [:,:,1] indicates rationale, 
        rationale = self.straight_through_sampling(gen_logits)
        # !debug
        #rationale = torch.zeros(rationale.shape).to(rationale.device)
        #rationale[:,:,1] = 1.0

        # mask rationale
        rationale = mask_ * rationale + (1.0 - mask_) * torch.cat(
            [all_ones, all_zeros], dim = -1
        )

        # ########## env inv predictor ##########
        env_inv_embeddings = mask_ * self.env_inv_embed_layer(text_ids)
        env_inv_rat_embeddings = env_inv_embeddings * rationale[:, :, 1].unsqueeze(dim = -1)
        env_inv_enc_outputs, _ = self.env_inv_encoder(env_inv_rat_embeddings)
        
        # max pooling and fc layer
        # make <pad> small to not influence the maxpooling operation
        env_inv_enc_outputs_ = mask_ * env_inv_enc_outputs + (1. - mask_) * (-1e-9)
        env_inv_enc_output, _ = torch.max(env_inv_enc_outputs_, dim=1) # (batch_size, emb_dim)
        env_inv_logits = self.env_inv_fc(env_inv_enc_output)

        # ########## env enable predictor ##########
        env_enable_embeddings = mask_ * self.env_enable_embed_layer(text_ids)
        env_enable_rat_embeddings = env_enable_embeddings * rationale[:, :, 1].unsqueeze(dim = -1)

        env_ = env.unsqueeze(dim = 1).float().expand(-1, mask.shape[1], -1) # (batch_size, seq_len, num_env)
        env_enable_enc_inputs = torch.cat([env_enable_rat_embeddings, env_], dim = -1)
        env_enable_enc_outputs, _ = self.env_enable_encoder(env_enable_enc_inputs)
        # max pooling and fc layer
        env_enable_enc_outputs = mask_ * env_enable_enc_outputs + (1. - mask_) * (-1e-9)
        env_enable_enc_output, _ = torch.max(env_enable_enc_outputs, dim = 1)
        env_enable_logits = self.env_enable_fc(env_enable_enc_output)

        return rationale, env_inv_logits, env_enable_logits
    
    def generator_trainable_variables(self):
        variables = []
        variables += list(self.gen_embed_layer.parameters())
        variables += list(self.generator.parameters())
        variables += list(self.generator_fc.parameters())
        
        return variables
    
    def env_inv_trainable_variables(self):
        variables = []
        variables += list(self.env_inv_embed_layer.parameters())
        variables += list(self.env_inv_encoder.parameters())
        variables += list(self.env_inv_fc.parameters())
        
        return variables

    def env_enable_trainable_variables(self):
        variables = []
        variables += list(self.env_enable_embed_layer.parameters())
        variables += list(self.env_enable_encoder.parameters())
        variables += list(self.env_enable_fc.parameters())
        
        return variables

class RNP_Bert(nn.Module):
    """
    Generator-predictor framework for rationalizing neural network. Lei 2016.

    Modules:
        gen_embed_layer: BertModel
        generator: GRU
        pre_embed_layer: BertModel
        dropout
        predictor_fc: nn.Linear
    """
    def __init__(
        self, 
        hidden_size,
        num_labels,
        bert_config = None,
        bert_path = None
    ):
        """
        Input:
            bert_config: a BertConfig obj or pretrained path
            bert_path: pretrained bert path
        """
        super().__init__()

        # Load bert encoder
        if bert_path is not None:
            self.gen_embed_layer = BertModel.from_pretrained(bert_path)
            self.pre_embed_layer = BertModel.from_pretrained(bert_path)
        elif isinstance(bert_config, BertConfig):
            self.gen_embed_layer = BertModel(bert_config)
            self.pre_embed_layer = BertModel(bert_config)
        elif isinstance(bert_config, str):
            self.gen_embed_layer = BertModel(BertConfig.from_pretrained(bert_config))
            self.pre_embed_layer = BertModel(BertConfig.from_pretrained(bert_config))
        else:
            raise ValueError(f"please specifiy bert_config or bert_path")
        config = self.gen_embed_layer.config
        emb_dim = config.hidden_size

        # initialize rational generator

        # version 1
        """
        self.generator = nn.GRU(
            emb_dim,
            hidden_size,
            num_layers = 1,
            bidirectional = True,
            batch_first = True
            )
        
        self.generator_fc = nn.Linear(hidden_size * 2, 2) # shape is 2 as mask or not mask
        """
        # version 2, remove gru
        self.generator_fc = nn.Linear(emb_dim, 2)

        # initialize sequence classifier
        self.dropout = nn.Dropout()
        self.predictor_fc = nn.Linear(config.hidden_size, num_labels)
    
    def straight_through_sampling(self, logits):
        """
        Input:
            logits -- (batch, seq_len, )
        """
        z = functional.softmax(logits, dim = -1)
        z_hard = functional.one_hot(torch.argmax(z, dim = -1), num_classes = z.shape[-1])
        # z_hard.requires_grads is False
        new_hard = z_hard - z.data + z

        return new_hard
        
    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
    ):
        """
        Inputs:
            input_ids -- (batch_size, seq_len)
            attention_mask -- (batch_size, seq_len)
        Outputs:
            rationale -- (batch_size, seq_len, 2)
            pred_logits -- (batch_size, num_class)
        """
        # aviod warning of RNN weights atorch
        #self.generator.flatten_parameters()
        
        # expand mask
        mask_ = attention_mask.unsqueeze(dim = -1)
        device = input_ids.device
        all_ones = torch.ones(input_ids.shape).unsqueeze(dim = -1).to(device)
        all_zeros = torch.zeros(all_ones.shape).to(device)

        # ########## generator ##########
        gen_bert_outputs = self.gen_embed_layer(input_ids, attention_mask, token_type_ids)
        gen_embeddings = mask_ * gen_bert_outputs[0]  # (batch, seq_len, hidden_size)

        # version 1
        #gen_outputs, _ = self.generator(gen_embeddings)
        #gen_logits = self.generator_fc(gen_outputs)

        # version 2
        gen_logits = self.generator_fc(gen_embeddings)

        # generate rationale (batch_size, seq_len, 2)
        # [:,:,1] indicates rationale, 
        rationale = self.straight_through_sampling(gen_logits)

        # mask rationale
        rationale = mask_ * rationale + (1.0 - mask_) * torch.cat(
            [all_ones, all_zeros], dim = -1
        )

        # ########## predictor ##########
        pred_bert_outputs = self.pre_embed_layer(input_ids, rationale[:,:, 1], token_type_ids)
        pooled_output = pred_bert_outputs[1]
        pooled_output = self.dropout(pooled_output)
        
        # max pooling and fc layer
        # make <pad> small to not influence the maxpooling operation
        #env_inv_enc_outputs_ = mask_ * env_inv_enc_outputs + (1. - mask_) * (-1e-9)
        #env_inv_enc_output, _ = torch.max(env_inv_enc_outputs_, dim=1) # (batch_size, emb_dim)
        
        # max pooling is conducted in theo model

        pred_logits = self.predictor_fc(pooled_output)

        return rationale, pred_logits
    
    def generator_trainable_variables(self):
        variables = []
        variables += list(self.gen_embed_layer.parameters())
        #variables += list(self.generator.parameters())
        variables += list(self.generator_fc.parameters())
        
        return variables
    
    def predictor_trainable_variables(self):
        variables = []
        variables += list(self.pre_embed_layer.parameters())
        variables += list(self.predictor_fc.parameters())
        
        return variables

if __name__ == '__main__':
    import numpy as np
    from utils import to_cuda

    emb_args = {
        'type': 'word2vec',
        "num_emb": 10,
        "emb_dim": 100,
        "pretrained_emb": np.random.rand(10, 100)
    }
    model = RNP_Bert(
        200,
        108,
        'bert-base-chinese',
        'bert-base-chinese'
    )

    model.cuda()
    
    #net = torch.nn.DataParallel(model)
    #print(net.module)
    #print(list(net.modules()))

    batch = {
        'input_ids': torch.randint(10, size = (8,50)),
        'attention_mask': torch.ones((8, 50), dtype = torch.int64),
        'token_type_ids': torch.zeros((8, 50), dtype = torch.int64),
    }
    batch = to_cuda(batch)
    rationale, pred_logits = model(**batch)

    print(f'rational: {rationale.shape}')
    print(f'pred_logits: {pred_logits.shape}')
    