# Models and related classes and functions
import torch
from torch.utils.data import Dataset
import transformers
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import numpy as np
import time
import math

class SingleSentenceEncoder_BERT(torch.nn.Module):
    """
    Single Sentence encoder based on a BERT kind encoder. 
    Single sentence means that each sentence is encoded in a 
    individual way, i.e, it doesn't take in account other sentences in the 
    same document.
    The sentence encoder must be a pre-trained model based on BERT architecture 
    like BERT, RoBERTa and ALBERT.
    """
    def __init__(self, encoder_id, embedding_dim):
        '''
        This model comprises only a pre-trained sentence encoder, which must be 
        a model following BERT architecture.  
        Arguments:
            encoder_id: ID (string) of the encoder model in Hugging Faces repository.
            embedding_dim: dimension of hidden units in the sentence encoder (e.g., 768 for BERT).
        '''
        super(SingleSentenceEncoder_BERT, self).__init__()
        self.encoder = transformers.AutoModel.from_pretrained(encoder_id)

    def forward(self, input_ids, attention_mask):
        '''
        Each call to this method encodes a batch of sentences. Each sentence is 
        individually encoded. This means the encoder doesn't take in account 
        other sentences from the source document when it encodes a sentence. 
        This method adopts the hidden state of the [CLS] token of the last BERT layer as 
        sentence representation and so it returns a batch of such representations.
        Arguments:
            input_ids : tensor of shape (batch_size, seq_len)
            attention_mask : tensor of shape (batch_size, seq_len)
        Returns:
            embeddings : tensor of shape (n of sentences in batch, embedding_dim)
        '''
        output = self.encoder(
            input_ids=input_ids,             # input_ids.shape: (batch_size, seq_len)
            attention_mask=attention_mask    # attention_mask.shape: (batch_size, seq_len)
        )
        hidden_state = output.last_hidden_state    # hidden states of last encoder's layer => shape: (batch_size, seq_len, embedding_dim)
        cls_embeddings = hidden_state[:, 0]        # hidden states of the CLS tokens from the last layer => shape: (batch_size, embedding_dim)

        return cls_embeddings

class SingleSC_BERT(torch.nn.Module):
    """
    Single Sentence Classifier based on a BERT kind encoder. 
    Single sentence means this classifier encodes each sentence in a 
    individual way, i.e, it doesn't take in account other sentences in the 
    same document.
    The sentence encoder must be a pre-trained model based on BERT architecture 
    like BERT, RoBERTa and ALBERT.
    """
    def __init__(self, encoder_id, n_classes, dropout_rate, embedding_dim):
        '''
        This model comprises a pre-trained sentence encoder and a classification head. 
        The sentence encoder must be a model following BERT architecture.  
        The classification head is a linear classifier (a single feedforward layer).
        Arguments:
            encoder_id: ID (string) of the encoder model in Hugging Faces repository.
            n_classes: number of classes.
            dropout_rate: dropout rate of classification layer.
            embedding_dim: dimension of hidden units in the sentence encoder (e.g., 768 for BERT).
        '''
        super(SingleSC_BERT, self).__init__()
        self.encoder = SingleSentenceEncoder_BERT(encoder_id, embedding_dim)
        dropout = torch.nn.Dropout(dropout_rate)
        n_classes = n_classes
        dense_out = torch.nn.Linear(embedding_dim, n_classes)
        torch.nn.init.xavier_uniform_(dense_out.weight)
        self.classifier = torch.nn.Sequential(dropout, dense_out)

    def forward(self, input_ids, attention_mask):
        '''
        Each call to this method process a batch of sentences. Each sentence is 
        individually encoded. This means the encoder doesn't take in account 
        other sentences from the source document when it encodes a sentence.
        This method returns one logit tensor for each sentence in the batch.
        Arguments:
            input_ids : tensor of shape (batch_size, seq_len)
            attention_mask : tensor of shape (batch_size, seq_len)
        Returns:
            logits : tensor of shape (n of sentences in batch, n of classes)
        '''
        '''
        output_1 = self.encoder(
            input_ids=input_ids,             # input_ids.shape: (batch_size, seq_len)
            attention_mask=attention_mask    # attention_mask.shape: (batch_size, seq_len)
        )
        hidden_state = output_1.last_hidden_state  # hidden states of last encoder's layer => shape: (batch_size, seq_len, embedd_dim)
        cls_embeddings = hidden_state[:, 0]        # hidden states of the CLS tokens from the last layer => shape: (batch_size, embedd_dim)
        '''
        cls_embeddings = self.encoder(
            input_ids=input_ids,             # input_ids.shape: (batch_size, seq_len)
            attention_mask=attention_mask    # attention_mask.shape: (batch_size, seq_len)
        )
        logits = self.classifier(cls_embeddings)   # logits.shape: (batch_size, num of classes)

        return logits

class MockSC_BERT(torch.nn.Module):
    '''
    A mock of SingleSC_BERT. It's usefull to accelerate the validation
    of the training loop.
    '''
    def __init__(self, n_classes):
        super(MockSC_BERT, self).__init__()
        self.classifier = torch.nn.Linear(10, n_classes) # 10 => random choice

    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.shape[0]
        mock_data = torch.rand((batch_size, 10), device=input_ids.device)
        logits = self.classifier(mock_data)    # shape: (batch_size, n_classes)

        return logits

class Single_SC_Dataset(Dataset):
    """
    A dataset object to be used together a SingleSC_BERT model. 
    Each item of the dataset represents a sole sentence.
    This object doesn't take in account the source documents of the 
    sentences.
    """
    def __init__(self, sentences, labels, labels_to_idx, tokenizer):
        """
        Arguments:
            sentences : list of strings.
            labels : list of strings.
            labels_to_idx : dictionary that maps each label (string) to a index (integer).
            tokenizer : the tokenizer to be used to split the sentences into inputs 
                of a SingleSC_BERT.
        """
        self.len = len(sentences)
        self.labels = list(labels)
        self.targets = []
        for l in labels:
            self.targets.append(labels_to_idx[l])
        self.targets = torch.tensor(self.targets, dtype=torch.long)
        self.data = tokenizer(
            sentences, 
            add_special_tokens=True,
            padding='longest', 
            return_token_type_ids=False, 
            return_attention_mask=True, 
            truncation=True, 
            max_length=512, 
            return_tensors='pt'
        )

    def __getitem__(self, index):
        return {
            'ids': self.data['input_ids'][index],       # shape: (seq_len)
            'mask': self.data['attention_mask'][index], # shape: (seq_len)
            'target': self.targets[index],              # shape: (1)
            'label': self.labels[index]                 # shape: (1)
        }
    
    def __len__(self):
        return self.len

def get_dataset(docs_dic, labels_to_idx, tokenizer):
    """
    Creates and returns a dataset for a set of documents.
    Arguments:
        docs_dic : a dictionary mapping document IDs/names to pandas Dataframes. Each 
            Dataframe must be the 'sentence' and 'label' columns.
        labels_to_idx : dictionary that maps each label (string) to a index (integer).
        tokenizer : the tokenizer to be used to split the sentences into inputs 
            of a SingleSC_BERT.
    Returns:
        An instance of Single_SC_Dataset.
    """
    sentences = []
    labels = []
    for _, df in docs_dic.items():
        df = df.drop(df[(df.label == 'None') | (df.label == 'Dissent')].index) # ignores None and Dissent labels
        sentences.extend(df['sentence'].to_list())
        labels.extend(df['label'].to_list())
    return Single_SC_Dataset(sentences, labels, labels_to_idx, tokenizer)

def count_labels(ds):
    """
    Returns the number of sentences by label for a provided Single_SC_Dataset.
    Arguments:
        ds : a Single_SC_Dataset.
    Returns:
       A dictionary mapping each label (string) to its number of sentences (integer).
    """
    count_by_label = {l: 0 for l in ds.labels}
    for l in ds.labels:
        count_by_label[l] = count_by_label[l] + 1
    return count_by_label

def evaluate(model, test_dataloader, loss_function, device):
    """
    Evaluates a provided SingleSC model.
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
            y_true_batch = data['target'].to(device)
            y_hat = model(ids, mask)
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
    Creates and train an instance of SingleSC_BERT.
    Arguments:
        train_params: dictionary storing the training params.
        ds_train: instance of Single_SC_Dataset storing the train data.
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
    
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=False)
    
    if use_mock:
        sentence_classifier = MockSC_BERT(n_classes).to(device)
    else:
        sentence_classifier = SingleSC_BERT(
            encoder_id, 
            n_classes, 
            dropout_rate, 
            embedding_dim
        ).to(device)
    
    if train_params['freeze_layers'] and not use_mock:
        sentence_classifier.encoder.requires_grad_(False)
    
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
            y_hat = sentence_classifier(ids, mask)
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
