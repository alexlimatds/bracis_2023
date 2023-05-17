# Models and related classes and functions
import time, math, torch, transformers
from torch.utils.data import Dataset
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import numpy as np
import singlesc_models

class PositionalEncoder:
    '''
    Computes a Sinusoidal Positional Encoder matrix.
    '''
    # Adapted from https://torchtutorialstaging.z5.web.core.windows.net/beginner/transformer_tutorial.html
    def __init__(self, embedding_dim, max_len=5000):
        '''
        Computes a PE matrix with shape (max_len, embedding_dim).
        Arguments:
            embedding_dim: the dimension of position vector.
            max_len: the maximum supported sequence lenght.
        '''
        super(PositionalEncoder, self).__init__()
        assert max_len <= 10000
        self.embedding_dim = embedding_dim
        self.pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
    
    def get_embeddings(self, seq_len):
        '''
        Returns a row subset of the PE matrix. The returned subset concerns a range of rows from 0 to seq_len - 1.
        '''
        return self.pe[0:seq_len]

class Content_PE_Dataset(torch.utils.data.Dataset):
    """
    A dataset object to be used together a SingleSC_PE_BERT model. 
    Each item of the dataset represents an inidividual sentence. The positional embedding of each sentence is also provided.
    """
    def __init__(self, dic_docs, labels_to_idx, tokenizer, max_seq_len, embedding_dim, positional_encoder):
        """
        Arguments:
            dic_docs: a dictionary whose each item is a document (key: docId, value: pandas DataFrame).
            labels_to_idx : dictionary that maps each label (string) to a index (integer).
            tokenizer : tokenizer of the encoder model.
            max_seq_len: maximum sequence length.
            embedding_dim: 
            positional_encoder: instance of PositionalEncoder.
        """
        self.input_ids = []             # tensor of shape (n_valid_sentences, max_seq_len)
        self.masks = []                 # tensor of shape (n_valid_sentences, max_seq_len)
        self.positional_embeddings = [] # tensor of shape (n_valid_sentences, embedding_dim)
        self.targets = []               # tensor of shape (n_valid_sentences)
        self.labels = []                # list of strings (n_valid_sentences)
        
        for df in dic_docs.values():
            doc_sentences = df['sentence'].tolist()
            doc_tk_data = tokenizer(
                doc_sentences, 
                add_special_tokens=True,
                padding='max_length', 
                return_token_type_ids=False, 
                return_attention_mask=True, 
                truncation=True, 
                max_length=max_seq_len, 
                return_tensors='pt'
            )
            
            doc_labels = df['label'].tolist()
            
            doc_targets = [labels_to_idx[l] for l in doc_labels]
            doc_targets = torch.tensor(doc_targets, dtype=torch.long)
            
            doc_pe = positional_encoder.get_embeddings(len(doc_labels))
            
            # letting only data regarding the valid sentences
            idx_valid = (doc_targets >= 0).nonzero().squeeze()
            self.input_ids.append(doc_tk_data['input_ids'][idx_valid])
            self.masks.append(doc_tk_data['attention_mask'][idx_valid])
            self.targets.append(doc_targets[idx_valid])
            self.positional_embeddings.append(doc_pe[idx_valid])
            for i in idx_valid:
                self.labels.append(doc_labels[i.item()])
            
        self.input_ids = torch.vstack(self.input_ids)
        self.masks = torch.vstack(self.masks)
        self.positional_embeddings = torch.vstack(self.positional_embeddings)
        self.targets = torch.hstack(self.targets)
        assert len(self.labels) == self.targets.shape[0]
        assert self.targets.shape[0] == self.input_ids.shape[0]
        assert self.input_ids.shape[0] == self.positional_embeddings.shape[0]

    def __getitem__(self, index):
        return {
            'ids': self.input_ids[index],            # PyTorch tensor with shape (max_seq_len)
            'mask': self.masks[index],               # PyTorch tensor with shape (max_seq_len)
            'pe': self.positional_embeddings[index], # PyTorch tensor with shape (embedding_dim)
            'target': self.targets[index],           # PyTorch tensor with shape (1)
            'label': self.labels[index]              # List with lenght (1)
        }
    
    def __len__(self):
        return len(self.labels)

class SingleSC_PE_BERT(torch.nn.Module):
    """
    Single Sentence Classifier based on a BERT kind encoder and on positional embeddings. 
    Single sentence means this classifier encodes each sentence in a 
    individual way.
    The sentence encoder must be a pre-trained model based on BERT architecture 
    like BERT, RoBERTa and ALBERT.
    """
    def __init__(self, encoder, n_classes, dropout_rate, embedding_dim, combination):
        '''
        This model comprises a pre-trained sentence encoder and a classification head. 
        The sentence encoder must be a model following BERT architecture.  
        The classification head is a linear classifier (a single feedforward layer).
        Arguments:
            encoder: an instance of singlesc_models.SingleSentenceEncoder_BERT.
            n_classes: number of classes.
            dropout_rate: dropout rate of the classification layer.
            embedding_dim: dimension of hidden units in the sentence encoder (e.g., 768 for BERT).
            combination: a string indicating how sentence embeddings and positional embeddings should 
                be combined. Employ S for sum and C for concatenation.
        '''
        super(SingleSC_PE_BERT, self).__init__()
        assert combination in ['S', 'C']
        self.encoder = encoder
        self.combination = combination
        classifier_dim = embedding_dim if combination == 'S' else embedding_dim * 2
        dropout = torch.nn.Dropout(dropout_rate)
        dense_out = torch.nn.Linear(classifier_dim, n_classes)
        torch.nn.init.xavier_uniform_(dense_out.weight)
        self.classifier = torch.nn.Sequential(dropout, dense_out)

    def forward(self, input_ids, attention_mask, pe):
        '''
        Each call to this method process a batch of sentences. Each sentence is 
        individually encoded. This means the encoder doesn't take in account 
        other sentences from the source document when it encodes a sentence.
        This method returns one logit tensor for each sentence in the batch.
        Arguments:
            input_ids : tensor of shape (batch_size, seq_len)
            attention_mask : tensor of shape (batch_size, seq_len)
            pe: positional embeddings. Tensor of shape (batch_size, seq_len, embedding_dim)
        Returns:
            logits : tensor of shape (n of sentences in batch, n_classes)
        '''
        cls_embeddings = self.encoder(
            input_ids=input_ids,             # input_ids.shape: (batch_size, seq_len)
            attention_mask=attention_mask    # attention_mask.shape: (batch_size, seq_len)
        )
        
        if self.combination == 'S':
            combined_embeddings = cls_embeddings + pe   # combined_embeddings.shape: (batch_size, embedding_dim)
        elif self.combination == 'C':
            combined_embeddings = torch.hstack((cls_embeddings, pe))  # combined_embeddings.shape: (batch_size, embedding_dim * 2)
        else:
            raise ValueError(f'Invalid value for self.combination: {self.combination}')
        
        logits = self.classifier(combined_embeddings)   # logits.shape: (batch_size, num of classes)

        return logits

class MockEncoder(torch.nn.Module):
    def __init__(self, embed_dim):
        super(MockEncoder, self).__init__()
        self.embedding_dim = embed_dim

    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.shape[0]
        mock_data = torch.rand((batch_size, self.embedding_dim), device=input_ids.device)

        return mock_data

def evaluate(model, test_dataloader, loss_function, device):
    """
    Evaluates a provided SingleSC_PE_BERT model.
    Arguments:
        model: the model to be evaluated.
        test_dataloader: torch.utils.data.DataLoader instance containing the test data.
        loss_function: instance of the loss function used to train the model.
        device: device where the model is located.
    Returns:
        eval_loss (float): the computed test loss score.
        precision (float): the computed test Precision score.
        recall (float): the computed test Recall score.
        f1 (float): the computed test F1 score.
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
            pe = data['pe'].to(device)
            y_true_batch = data['target'].to(device)
            y_hat = model(ids, mask, pe)
            loss = loss_function(y_hat, y_true_batch)
            eval_loss += loss.item()
            predictions_batch = y_hat.argmax(dim=1)
            predictions = torch.cat((predictions, predictions_batch))
            y_true = torch.cat((y_true, y_true_batch))
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
    Creates and train an instance of SingleSC_PE_BERT.
    Arguments:
        train_params: dictionary storing the training params.
        ds_train: instance of Single_SC_Dataset storing the train data.
        ds_test: instance of Single_SC_Dataset storing the test data.
        tokenizer: the tokenizer of the chosen pre-trained sentence encoder.
        device: device where the model should be located.
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
    combination = train_params['combination']
    
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=False)
    
    if use_mock:
        sentence_encoder = MockEncoder(embedding_dim).to(device)
    else:
        sentence_encoder = singlesc_models.SingleSentenceEncoder_BERT(encoder_id, embedding_dim).to(device)
        
    sentence_classifier = SingleSC_PE_BERT(
        sentence_encoder, 
        n_classes, 
        dropout_rate, 
        embedding_dim, 
        combination
    ).to(device)
    
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(
        sentence_classifier.parameters(), 
        lr=learning_rate, 
        betas=(0.9, 0.999), 
        eps=1e-8, 
        weight_decay=weight_decay
    )
    num_training_steps = len(dl_train) * n_epochs
    lr_scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps = 0, 
        num_training_steps = num_training_steps
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
            pe = train_data['pe'].to(device)
            y_hat = sentence_classifier(ids, mask, pe)
            y_true = train_data['target'].to(device)
            loss = criterion(y_hat, y_true)
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
