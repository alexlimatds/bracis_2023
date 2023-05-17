# Models and related classes and functions
import torch
from torch.utils.data import Dataset
from torch.distributions.beta import Beta
import transformers
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import numpy as np
import time, math, random
import singlesc_models, mixup_models


def evaluate(model, test_dataloader, loss_function, device):
    """
    Evaluates a provided MixupSentenceClassifier model.
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
            y_hat = model(ids, mask)
            y_true_batch = data['target'].to(device)
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

def fit(train_params, ds_train, ds_test, labels_to_idx, device):
    """
    Creates and train an instance of singlesc_models.SingleSC_BERT. The training leverages 
    mixup data.
    Arguments:
        train_params: dictionary storing the training params.
        ds_train: instance of MixupSentenceClassifier_Dataset storing the train data.
        ds_test: instance of MixupSentenceClassifier_Dataset storing the test data.
        labels_to_idx: dictionary mapping labels (keys) to class IDs (values).
        device: device where the model should be located.
    """
    learning_rate = train_params['learning_rate']
    weight_decay = train_params['weight_decay']
    n_epochs = train_params['n_epochs']
    batch_size = train_params['batch_size']
    n_classes = train_params['n_classes']
    dropout_rate = train_params['dropout_rate']
    embedding_dim = train_params['embedding_dim']
    use_mock = train_params['use_mock']
    
    # Mixup params
    alpha = train_params['mixup_alpha']
    augmentation_rate = train_params['augmentation_rate']
    classes_to_augment = torch.tensor(
        [labels_to_idx[l] for l in train_params['classes_to_augment']], 
        dtype=torch.long
    )
    classes_to_augment_one_hot = torch.nn.functional.one_hot(classes_to_augment, num_classes=n_classes).type(torch.float).to(device)
    
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=False)
    
    if use_mock:
        sentence_classifier = mixup_models.MockSC_BERT(n_classes, embedding_dim).to(device)
    else:
        sentence_classifier = singlesc_models.SingleSC_BERT(
            train_params['encoder_id'], 
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
    n_augmented_vectors = [] # it will store the mean number of synthetic vectors by epoch
    start_train = time.perf_counter()
    for epoch in range(1, n_epochs + 1):
        print(f'  Starting epoch {epoch}... ', end='')
        start_epoch = time.perf_counter()
        epoch_loss = 0
        n_augmented_epoch = 0
        sentence_classifier.train()
        for train_data in dl_train:
            optimizer.zero_grad()
            # encoding sentences
            ids = train_data['ids'].to(device)
            mask = train_data['mask'].to(device)
            x = sentence_classifier.encoder(ids, mask)
            # converting targets to one-hot encoded vectors
            y_true = train_data['target'].to(device)
            y_true_one_hot = torch.nn.functional.one_hot(y_true, num_classes=n_classes).type(torch.float).to(device)
            # generating mixup data
            x_mixup, y_mixup = mixup_models.augment_data(
                alpha, 
                x, 
                y_true_one_hot, 
                classes_to_augment_one_hot, 
                train_params['augmentation_rate']
            )
            # joining data (should we shuffle the joined set?)
            if (x_mixup is not None) and (y_mixup is not None):
                x_j = torch.vstack([x, x_mixup])
                y_j = torch.vstack([y_true_one_hot, y_mixup])
                n_augmented_epoch += x_mixup.shape[0]
            else:
                x_j = x
                y_j = y_true_one_hot
            # classification
            y_hat = sentence_classifier.classifier(x_j)
            # loss and backward
            loss = criterion(y_hat, y_j)
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
        n_augmented_vectors.append(n_augmented_epoch)
        
        end_epoch = time.perf_counter()
        print('finished! Time: ', time.strftime("%Hh%Mm%Ss", time.gmtime(end_epoch - start_epoch)))
    
    n_augmented_vectors = np.array(n_augmented_vectors).mean()
    end_train = time.perf_counter()
    
    return metrics, confusion_matrices, time.strftime("%Hh%Mm%Ss", time.gmtime(end_train - start_train)), n_augmented_vectors
