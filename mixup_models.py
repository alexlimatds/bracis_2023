# Models and related classes and functions
import torch
from torch.utils.data import Dataset
from torch.distributions.beta import Beta
import transformers
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import numpy as np
import time, math, random
import singlesc_models

class SentenceClassifier(torch.nn.Module):
    """
    Sentence Classifier with one linear layer. 
    """
    def __init__(self, n_classes, dropout_rate, embedding_dim):
        '''
        This model comprises only a classification head, which is a linear classifier (a single feedforward layer).
        Arguments:
            n_classes: number of classes.
            dropout_rate: dropout rate of classification layer.
            embedding_dim: dimension of input vectors.
        '''
        super(SentenceClassifier, self).__init__()
        dropout = torch.nn.Dropout(dropout_rate)
        
        dense_out = torch.nn.Linear(embedding_dim, n_classes)
        torch.nn.init.xavier_uniform_(dense_out.weight)
        self.classifier = torch.nn.Sequential(dropout, dense_out)

    def forward(self, input_data):
        '''
        Each call to this method process a batch of sentences.
        This method returns one logit tensor for each sentence in the batch (batch_size = number of sentences in the batch).
        Arguments:
            input_data : tensor of shape (batch_size, embedding_dim)
        Returns:
            logits : tensor of shape (batch_size, n_classes)
        '''
        logits = self.classifier(input_data)   # logits.shape: (batch_size, num of classes)

        return logits

class MockEncoder(torch.nn.Module):
    def __init__(self, embed_dim):
        super(MockEncoder, self).__init__()
        self.embedding_dim = embed_dim

    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.shape[0]
        mock_data = torch.rand((batch_size, self.embedding_dim), device=input_ids.device)

        return mock_data
    
class MockSC_BERT(torch.nn.Module):
    '''
    A mock of SingleSC_BERT. It's usefull to accelerate the validation
    of the training loop.
    '''
    def __init__(self, n_classes, embed_dim):
        super(MockSC_BERT, self).__init__()
        self.encoder = MockEncoder(embed_dim)
        self.classifier = torch.nn.Linear(embed_dim, n_classes)

    def forward(self, input_ids, attention_mask):
        mock_data = self.encoder(input_ids, attention_mask)
        logits = self.classifier(mock_data)    # shape: (batch_size, n_classes)

        return logits
    
class SentenceClassifier_Dataset(Dataset):
    """
    A dataset object to be used together a SentenceClassifier model. 
    Each item of the dataset represents a sole sentence embedding.
    This object doesn't take in account the source documents of the 
    sentences.
    """
    def __init__(self, sent_embeddings, targets):
        """
        Arguments:
            sent_embeddings : a PyTorch tensor with shape (n_sentences, embedd_dim) 
                representing the sentence embeddings.
            targets : a PyTorch tensor with shape (n_sentences, n_classes) 
                representing the one-hot encoded targets.
        """
        self.embeddings = sent_embeddings
        self.targets = targets

    def __getitem__(self, index):
        return {
            'embeddings': self.embeddings[index],  # shape: (embedding_dim)
            'target': self.targets[index]          # shape: (n_classes)
        }
    
    def __len__(self):
        return self.embeddings.shape[0]


def evaluate(model, test_dataloader, loss_function, device):
    """
    Evaluates a provided SentenceClassifier model.
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
            embeddings = data['embeddings'].to(device)
            y_hat = model(embeddings)
            y_true_batch = data['target'].to(device)
            loss = loss_function(y_hat, y_true_batch)
            eval_loss += loss.item()
            predictions_batch = y_hat.argmax(dim=1)
            predictions = torch.cat((predictions, predictions_batch))
            y_true = torch.cat((
                y_true, 
                y_true_batch.argmax(dim=1)
            ))
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
    Creates and train an instance of SentenceClassifier.
    Arguments:
        train_params: dictionary storing the training params.
        ds_train: instance of SentenceClassifier_Dataset storing the train data.
        ds_test: instance of SentenceClassifier_Dataset storing the test data.
        device: device where the model is located.
    """
    learning_rate = train_params['learning_rate_classifier']
    weight_decay = train_params['weight_decay']
    n_epochs = train_params['n_epochs_classifier']
    batch_size = train_params['batch_size']
    n_classes = train_params['n_classes']
    dropout_rate = train_params['dropout_rate']
    embedding_dim = train_params['embedding_dim']
    
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=False)
    
    sentence_classifier = SentenceClassifier(
        n_classes, 
        dropout_rate, 
        embedding_dim
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
        print(f'  Starting epoch {epoch}... ', end='')
        start_epoch = time.perf_counter()
        epoch_loss = 0
        sentence_classifier.train()
        for train_data in dl_train:
            optimizer.zero_grad()
            embeddings = train_data['embeddings'].to(device)
            y_hat = sentence_classifier(embeddings)
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

def mixup(xi, xj, yi, yj, alpha):
    """
    Mixup function: generates a synthetic vector from two source vectors. For details, 
    check the mixup paper.
    Arguments:
        xi : the first source vectors. PyTorch tensor of shape (n_sentences, embedding_dim)
        xj : the second source vectors. PyTorch tensor of shape (n_sentences, embedding_dim)
        yi : the hot-one-encoded label vector of the first source vectors. PyTorch tensor of shape (n_sentences, n_classes)
        yj : the hot-one-encoded label vector of the second source vectors. PyTorch tensor of shape (n_sentences, n_classes)
        alpha : hyperparameter of the beta distribution to be used to generate the lambda value.
    Returns:
        The generated synthetic vectors. PyTorch tensor of shape (n_sentences, embedding_dim)
        The targets vectors of the generated synthetic vectors. PyTorch tensor of shape (n_sentences, n_classes)
    """
    b = Beta(alpha, alpha)
    lam = b.rsample(sample_shape=(xi.shape[0], 1))
    lam_x = lam.broadcast_to(xi.shape).to(xi.device)
    x_hat = lam_x * xi + (1 - lam_x) * xj
    lam_y = lam.broadcast_to(yi.shape).to(yi.device)
    y_hat = lam_y * yi + (1 - lam_y) * yj
    return x_hat, y_hat

def augment_data(alpha, X, Y, classes_to_agument, augmentation_rate):
    """
    Generates a set of synthetic embedding vectors from the sentences in a provided set of 
    documents. The sentences are selected at random. It uses sentences of different 
    classes to generate a synthetic vector.
    Params:
        alpha: hyperparameter of the beta distribution to be used with the mixup algorithm.
        X : embedding vectors. PyTorch tensor of shape (n_sentences, embedding_dim).
        Y : one-hot encoded target vectors. PyTorch tensor of shape (n_sentences, n_classes).
        classes_to_agument: One-hot vectors of the classes whose vectors will be used to augment data. PyTorch tensor of shape (n classes to agument, n_classes).
        augmentation_rate: rate employed to calculate the number of augmented vectors for each class. Float.
    Returns:
        A tuple:
            The generated feature vectors. PyTorch tensor of shape (n_generated_vectors, embedding_dim).
            The generated target vectors PyTorch tensor of shape (n_generated_vectors, n_classes).
        If Y does not include instances of the classes to augment, it returns (None, None).
    """
    X_aug, Y_aug = [], []
    n_classes = Y.shape[1]
    for idx_target_class in classes_to_agument:
        # selecting vectors to support data augmentation
        n_synthetic = math.ceil(
            augmentation_rate * (Y == idx_target_class).all(dim=1).count_nonzero().item()
        )
        idx_target = (Y == idx_target_class).all(dim=1).nonzero().squeeze().tolist()
        if isinstance(idx_target, list) and len(idx_target) > 0:
            idx_other = [i for i in range(Y.shape[0]) if i not in idx_target]
            if len(idx_other) > 0:
                idx_i = random.sample(idx_target, k=n_synthetic) if n_synthetic <= len(idx_target) else random.choices(idx_target, k=n_synthetic) # random indexes for the target class
                idx_j = random.sample(idx_other, k=n_synthetic)  if n_synthetic <= len(idx_other)  else random.choices(idx_other, k=n_synthetic)  # random indexes for other classes
                # getting source vectors to generate the augmented vectors
                # target class
                x_i = X[idx_i, :]
                y_i = Y[idx_i, :]
                # other classes
                x_j = X[idx_j, :]
                y_j = Y[idx_j, :]
                # data augmentation
                x_hat, y_hat = mixup(x_i, x_j, y_i, y_j, alpha)
                X_aug.append(x_hat)
                Y_aug.append(y_hat)
    if len(X_aug) > 0:
        X_aug = torch.vstack(X_aug)
        Y_aug = torch.vstack(Y_aug)
    else:
        X_aug = None
        Y_aug = None
    
    return X_aug, Y_aug
