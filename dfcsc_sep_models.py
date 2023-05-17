# Models and related classes and functions
import torch, transformers
from torch.utils.data import Dataset
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import numpy as np
import time, dfcsc_cls_models

class DFCSC_SEP_BERT_Encoder(torch.nn.Module):
    """
    Sentence encoder that relys on a pre-trained model based on BERT's architecture and 
    that expects inputs following the DFCSC layout.
    """
    def __init__(self, encoder_id, sep_token_id):
        '''
        Arguments:
            encoder_id: ID (string) of the encoder model in Hugging Faces repository.
            sep_token_id: ID (integer) of the [SEP] token.
        '''
        super(DFCSC_SEP_BERT_Encoder, self).__init__()
        self.sep_token_id = sep_token_id
        self.encoder = transformers.AutoModel.from_pretrained(encoder_id)

    def forward(self, input_ids, attention_mask):
        '''
        Encodes a batch of DFCSCs. The model exploits the edge tokens and the core sentences to 
        produce embeddings for the core sentences, but it doesn't produce embeddings for edge tokens.
        This method returns one logit tensor for each core sentence in the batch.
        Arguments:
            input_ids : tensor of shape (batch_size, seq_len)
            attention_mask : tensor of shape (batch_size, seq_len)
        Returns:
            PyTorch tensor with shape (n core sentences in batch, embedding_dim). This tensor 
            holds the embeddings of the core sentences from all chunks in the batch. Each embedding 
            is the respective last hidden state of the [SEP] token of the respective core sentence.
        '''
        output_1 = self.encoder(
            input_ids=input_ids,             # input_ids.shape: (batch_size, seq_len)
            attention_mask=attention_mask    # attention_mask.shape: (batch_size, seq_len)
        )
        hidden_state = output_1.last_hidden_state  # hidden states of last encoder's layer => shape: (batch_size, seq_len, embedd_dim)
        
        ### In this approach we rely on [SEP] embeddings to represent sentences
        # getting embeddings of core [SEP] tokens
        batch_size = input_ids.shape[0]
        core_embeddings = [] # core_embeddings.shape: (n core sentences in batch, embedding_dim)
        for i in range(batch_size): # iterates chunks
            idx_seps = torch.nonzero(input_ids[i] == self.sep_token_id, as_tuple=True)[0] # indexes of all [SEP] tokens in currrent sentence
            # we dont want the embeddings of the 1st and last [SEP] tokens
            idx_core_sep = idx_seps[1:-1] # index of core [SEP] tokens in current chunk
            core_sep_emb = hidden_state[i, idx_core_sep, :] # embeddings of core [SEP]. shape: (n core sentences in chunk, embedd_dim)
            core_embeddings.append(core_sep_emb)
        core_embeddings = torch.vstack(core_embeddings) # core_embeddings.shape: (n core sentences in batch, embedding_dim)
        
        return core_embeddings

class DFCSC_SEP_LongformerEncoder(torch.nn.Module):
    '''
    Uses a Longformer model to encode DFCSCs.
    '''
    def __init__(self, encoder_id, sep_token_id):
        '''
        Arguments:
            encoder_id: ID (string) of the encoder model in Hugging Faces repository.
            sep_token_id: ID (integer) of the </s> token.
        '''
        super(DFCSC_SEP_LongformerEncoder, self).__init__()
        self.sep_token_id = sep_token_id
        self.encoder = transformers.AutoModel.from_pretrained(encoder_id)
        
    def forward(self, input_ids, attention_mask):
        '''
        Encodes a batch of DFCSCs.
        Arguments:
            input_ids: PyTorch tensor with shape (batch_size, chunk_len).
            attention_mask: PyTorch tensor with shape (batch_size, chunk_len).
        Returns:
            PyTorch tensor with shape (n core sentences in batch, embedding_dim). This tensor 
            holds the embeddings of the core sentences from all chunks in the batch. Each embedding 
            is the respective last hidden state of the </s> token of the respective core sentence.
        '''
        # The global_attention_mask tells which tokens must have global attention. 
        # Here, <s> and </s> have global attention
        global_attention_mask = torch.zeros(
            input_ids.shape, 
            dtype=torch.long, 
            device=input_ids.device
        )
        global_attention_mask[:, 0] = 1 # global attention for <s> since it must be the first token in a chunk
        idx_sep = torch.nonzero(input_ids == self.sep_token_id, as_tuple=True)
        for i in range(idx_sep[0].shape[0]):
            global_attention_mask[idx_sep[0][i], idx_sep[1][i]] = 1 # global attention for </s>
        
        # Encoding
        output_1 = self.encoder(
            input_ids=input_ids,             # input_ids.shape: (batch_size, chunk_len)
            attention_mask=attention_mask    # attention_mask.shape: (batch_size, chunk_len)
        )
        hidden_state = output_1.last_hidden_state  # hidden states of last encoder's layer => shape: (batch_size, chunk_len, embedd_dim)
        
        ### In this approach we rely on </s> embeddings to represent sentences
        # getting embeddings of core </s> tokens
        batch_size = input_ids.shape[0]
        core_embeddings = []
        '''
        for i in range(batch_size): # iterates chunks
            cls_embedding = hidden_state[i, 0, :] # cls_embedding.shape: (embedd_dim)
            idx_seps = torch.nonzero(input_ids[i] == self.sep_token_id, as_tuple=True)[0] # indexes of all </s> tokens in currrent sentence
            # we dont want the embeddings of the 1st and last </s> tokens
            idx_core_sep = idx_seps[1:-1] # index of core </s> tokens in current chunk
            core_sep_emb = hidden_state[i, idx_core_sep, :] # embeddings of core </s>. shape: (n core sentences in chunk, embedd_dim)
            for j in range(core_sep_emb.shape[0]):
                core_embeddings.append(
                    torch.hstack((cls_embedding, core_sep_emb[j]))
                )
        core_embeddings = torch.vstack(core_embeddings) # core_embeddings.shape: (n core sentences in batch, embedding_dim * 2)
        '''
        #
        for i in range(batch_size): # iterates chunks
            idx_seps = torch.nonzero(input_ids[i] == self.sep_token_id, as_tuple=True)[0] # indexes of all </s> tokens in currrent sentence
            # we dont want the embeddings of the 1st and last </s> tokens
            idx_core_sep = idx_seps[1:-1] # index of core </s> tokens in current chunk
            core_sep_emb = hidden_state[i, idx_core_sep, :] # embeddings of core </s>. shape: (n core sentences in chunk, embedd_dim)
            core_embeddings.append(core_sep_emb)
        core_embeddings = torch.vstack(core_embeddings) # core_embeddings.shape: (n core sentences in batch, embedding_dim)
        
        return core_embeddings

class DFCSC_SEP_Classifier(torch.nn.Module):
    """
    Sentence Classifier based on core [SEP] embeddings from a DFCSC.
    """
    def __init__(self, encoder, n_classes, dropout_rate, embedding_dim):
        '''
        This model comprises a sentence encoder and a classification head. 
        The sentence encoder must be a model that yelds [SEP] (or </s>) embeddings from 
        inputs following a DFCSC layout.
        The classification head is a linear classifier (a single feedforward layer).
        Arguments:
            encoder: the sentence encoder.
            n_classes: number of classes.
            dropout_rate: dropout rate of classification layer.
            embedding_dim: dimension of hidden units in the sentence encoder (e.g., 768 for Longformer).
        '''
        super(DFCSC_SEP_Classifier, self).__init__()
        self.encoder = encoder
        
        dropout = torch.nn.Dropout(dropout_rate)
        dense_out = torch.nn.Linear(embedding_dim, n_classes)
        torch.nn.init.xavier_uniform_(dense_out.weight)
        self.classifier = torch.nn.Sequential(
            dropout, dense_out
        )

    def forward(self, input_ids, attention_mask):
        '''
        Each call to this method process a batch of DFCSCs. The model exploits the 
        edge tokens and the core sentences to produce embeddings for the core sentences, 
        but it doesn't produce embeddings for edge tokens.
        This method returns one logit tensor for each core sentence in the batch.
        Arguments:
            input_ids : tensor of shape (batch_size, seq_len)
            attention_mask : tensor of shape (batch_size, seq_len)
        Returns:
            logits : tensor of shape (n of core sentences in batch, n of classes)
        '''
        core_embeddings = self.encoder(input_ids, attention_mask)
        
        logits = self.classifier(core_embeddings)   # logits.shape: (n core sentences in batch, num of classes)

        return logits

class MockDFCSC_SEP_Encoder(torch.nn.Module):
    '''
    A mock of sentence encoder based on DFCSCs. It's usefull to accelerate the validation
    of the training loop.
    '''
    def __init__(self, sep_token_id, embedding_dim):
        super(MockDFCSC_SEP_Encoder, self).__init__()
        self.sep_token_id = sep_token_id
        self.embedding_dim = embedding_dim

    def forward(self, input_ids, attention_mask):
        # getting number of labels/targets
        n_chunks = input_ids.shape[0]
        n_sep_tokens = torch.count_nonzero(input_ids == self.sep_token_id).item()
        n_targets = n_sep_tokens - 2 * n_chunks # -2 because we ignore [CLS] token and the las [SEP] token
        
        return torch.rand((n_targets, self.embedding_dim), device=input_ids.device)
    
def evaluate(model, test_dataloader, loss_function, device):
    """
    Evaluates a provided DFCSC_SEP_Classifier model.
    Arguments:
        model: the model to be evaluated.
        test_dataloader: torch.utils.data.DataLoader instance containing the test data.
        loss_function: instance of the loss function used to train the model.
        device: device where the model is located.
    Returns:
        eval_loss (float): the computed test loss score.
        precision (float): the computed test macro Precision score.
        recall (float): the computed test macro Recall score.
        f1 (float): the computed test macro F1 score.
        confusion_matrix: the computed test confusion matrix.
    """
    predictions = torch.tensor([]).to(device)
    y_true = torch.tensor([]).to(device)
    eval_loss = 0
    model.eval()
    with torch.no_grad():
        for data in test_dataloader:
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            y_true_batch = data['targets'].to(device)
            y_hat = model(ids, mask)
            # ignores classes with negative ID
            idx_valid = (y_true_batch >= 0).nonzero().squeeze()
            y_true_batch_valid = y_true_batch[idx_valid]
            y_hat_valid = y_hat[idx_valid]
            loss = loss_function(y_hat_valid, y_true_batch_valid)
            eval_loss += loss.item()
            predictions_batch = y_hat_valid.argmax(dim=1)
            predictions = torch.cat((predictions, predictions_batch))
            y_true = torch.cat((y_true, y_true_batch_valid))
        predictions = predictions.detach().to('cpu').numpy()
        y_true = y_true.detach().to('cpu').numpy()
    eval_loss = eval_loss / len(test_dataloader)
    t_metrics_macro = precision_recall_fscore_support(
        y_true, 
        predictions, 
        average='macro', 
        zero_division=0
    )
    cm = confusion_matrix(
        y_true, 
        predictions
    )
    
    return eval_loss, t_metrics_macro[0], t_metrics_macro[1], t_metrics_macro[2], cm

def fit(train_params, ds_train, ds_test, device):
    """
    Creates and train an instance of DFCSC_Classifier.
    Arguments:
        train_params: dictionary storing the training params.
        ds_train: instance of dfcsc_cls_models.DFCSC_Dataset storing the train data.
        tokenizer: the tokenizer of the chosen pre-trained sentence encoder.
        device: device where the model is located.
    """
    learning_rate = train_params['learning_rate']
    weight_decay = train_params['weight_decay']
    n_epochs = train_params['n_epochs']
    batch_size = train_params['batch_size']
    encoder_id = train_params['encoder_id']
    n_classes = train_params['n_classes']
    dropout_rate = train_params['dropout_rate']
    embedding_dim = train_params['embedding_dim']
    use_mock = train_params['use_mock']
    
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, collate_fn=dfcsc_cls_models.collate_batch)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=True, collate_fn=dfcsc_cls_models.collate_batch)
    
    # creating encoder
    if use_mock:
        encoder = MockDFCSC_SEP_Encoder(train_params['sep_token_id'], embedding_dim).to(device)
    elif encoder_id.lower().find('longformer') > -1:
        encoder = DFCSC_SEP_LongformerEncoder(encoder_id, train_params['sep_token_id']).to(device)
    else:
        encoder = DFCSC_SEP_BERT_Encoder(encoder_id, train_params['sep_token_id']).to(device)
    
    sentence_classifier = DFCSC_SEP_Classifier(encoder, n_classes, dropout_rate, embedding_dim).to(device)
    
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(
        sentence_classifier.parameters(), 
        lr=learning_rate, 
        betas=(0.9, 0.999), 
        eps=train_params['eps'], 
        weight_decay=weight_decay
    )
    lr_scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps = 0, 
        num_training_steps = len(dl_train) * n_epochs
    )
    
    metrics = {} # key: epoch number, value: numpy tensor storing train loss, test loss, Precision (macro), Recall (macro), F1 (macro)
    confusion_matrices = {} # key: epoch number, value: scikit-learn's confusion matrix
    start_train = time.perf_counter()
    for epoch in range(1, n_epochs + 1):
        print(f'Starting epoch {epoch}... ', end='')
        start_epoch = time.perf_counter()
        epoch_loss = 0
        sentence_classifier.train()
        for train_data in dl_train:
            optimizer.zero_grad()
            ids = train_data['ids'].to(device)
            mask = train_data['mask'].to(device)
            y_hat = sentence_classifier(ids, mask)
            y_true = train_data['targets'].to(device)
            # ignores classes with negative ID
            idx_valid = (y_true >= 0).nonzero().squeeze()
            y_true_valid = y_true[idx_valid]
            y_hat_valid = y_hat[idx_valid]
            loss = criterion(y_hat_valid, y_true_valid)
            epoch_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sentence_classifier.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
        epoch_loss = epoch_loss / len(dl_train)
        # evaluation
        optimizer.zero_grad()
        eval_loss, p_macro, r_macro, f1_macro, cm = evaluate(
            sentence_classifier, 
            dl_test, 
            criterion, 
            device
        )
        #storing metrics
        metrics[epoch] = np.array([epoch_loss, eval_loss, p_macro, r_macro, f1_macro])
        confusion_matrices[epoch] = cm
        end_epoch = time.perf_counter()
        print('finished! Time: ', time.strftime("%Hh%Mm%Ss", time.gmtime(end_epoch - start_epoch)))
            
    end_train = time.perf_counter()
    
    return metrics, confusion_matrices, time.strftime("%Hh%Mm%Ss", time.gmtime(end_train - start_train))
