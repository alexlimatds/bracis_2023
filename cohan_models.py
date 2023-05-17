import torch
from torch.utils.data import Dataset
import numpy as np
import transformers, time
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# This model is used only when validating the train-evaluation-report workflow as way to accelerate it
class MockModel(torch.nn.Module):
    def __init__(self, sep_id, n_classes):
        super(MockModel, self).__init__()
        self.classifier = torch.nn.Linear(10, n_classes) # 10 => random value
        self.sep_id = sep_id
        self.n_classes = n_classes

    def forward(self, input_ids, attention_mask):
        idx_sep = input_ids.detach().to('cpu').flatten().numpy()
        idx_sep = np.nonzero(idx_sep == self.sep_id)[0]
        batch_size = idx_sep.shape[0] # batch_size equals to num of SEP tokens
        mock_data = torch.rand((batch_size, 10), device=input_ids.device)
        logits = self.classifier(mock_data)  # shape: (num of SEP tokens, n_classes)

        return logits
    
def evaluate(model, test_dataloader, loss_function, train_params, device):
    """
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
    chunk_layout = train_params['chunk_layout']
    predictions = torch.tensor([]).to(device)
    y_true = torch.tensor([]).to(device)
    eval_loss = 0
    model.eval()
    with torch.no_grad():
        for data in test_dataloader:
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            y_true_batch = data['targets'].to(device)
            if chunk_layout == 'Cohan':
                y_hat = model(ids, mask)
            elif chunk_layout == 'VanBerg':
                y_hat = model(ids, mask, data['n_overlaped_sentences'])
            idx_valid = (y_true_batch >= 0).nonzero().squeeze() # ignores classes with negative ID
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

def fit(train_params, ds_train, ds_test, labels_to_idx, device):
    """
    Creates and train a model
    Arguments:
        train_params: dictionary storing the training params.
        ds_train: instance of dataset storing the train data.
        ds_test: instance of dataset storing the test data.
        labels_to_idx: 
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
    chunk_layout = train_params['chunk_layout']
    model_reference = train_params['model_reference']
    
    if chunk_layout == 'Cohan':
        collate_fn = collate_batch_cohan
    elif chunk_layout == 'VanBerg':
        collate_fn = collate_OverlapedChunksDataset
    else:
        raise ValueError('Not supported chunk layout:', chunk_layout)
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    if use_mock:
        sentence_classifier = MockModel(train_params['sep_token_id'], n_classes).to(device)
    else:
        if chunk_layout == 'Cohan':
            if model_reference == 'Longformer':
                sentence_classifier = SSClassifier_Cohan_Longformer(
                    encoder_id, 
                    n_classes, 
                    dropout_rate, 
                    train_params['sep_token_id'], 
                    embedding_dim
                ).to(device)
            else: # model based on BERT architecture
                sentence_classifier = SSClassifier_Cohan(
                    encoder_id, 
                    n_classes, 
                    dropout_rate, 
                    train_params['sep_token_id'], 
                    embedding_dim
                ).to(device)
        elif chunk_layout == 'VanBerg':
            if model_reference == 'Longformer':
                sentence_classifier = SSClassifier_VanBerg_Longformer(
                    encoder_id, 
                    n_classes, 
                    dropout_rate, 
                    train_params['sep_token_id'], 
                    embedding_dim
                ).to(device)
            else: # model based on BERT architecture
                sentence_classifier = SSClassifier_VanBerg(
                    encoder_id, 
                    n_classes, 
                    dropout_rate, 
                    train_params['sep_token_id'], 
                    embedding_dim
                ).to(device)
    
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
    best_score = None
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
            if chunk_layout == 'Cohan':
                y_hat = sentence_classifier(ids, mask)
            elif chunk_layout == 'VanBerg':
                y_hat = sentence_classifier(ids, mask, train_data['n_overlaped_sentences'])
            y_true = train_data['targets'].to(device)
            idx_valid = (y_true >= 0).nonzero().squeeze() # ignores classes with negative ID
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
            train_params, 
            device
        )
        #storing metrics
        metrics[epoch] = np.array([epoch_loss, eval_loss, p_macro, r_macro, f1_macro])
        confusion_matrices[epoch] = cm
        end_epoch = time.perf_counter()
        print('finished! Time: ', time.strftime("%Hh%Mm%Ss", time.gmtime(end_epoch - start_epoch)))
            
    end_train = time.perf_counter()
    
    return metrics, confusion_matrices, time.strftime("%Hh%Mm%Ss", time.gmtime(end_train - start_train))

# **************** Isolated chunks ****************

class SSClassifier_Cohan(torch.nn.Module):
    def __init__(self, encoder_id, n_classes, dropout_rate, sep_id, embedding_dim):
        '''
        Creates a classifier of sequential sentences as proposed in the Pretrained Language Models 
        for Sequential Sentence Classification paper (Cohan et al, 2019). This model comprises a pre-trained sentence 
        encoder and a classification head. The sentence encoder must be a model following BERT architecture 
        such as BERT, RoBERTa, or ALBERT. The classification head is single fully-connected layer.
        Arguments:
            encoder_id: ID (string) of the encoder model in Hugging Faces repository.
            n_classes: number of classes.
            dropout_rate: dropout rate of classification layers.
            sep_id: the ID of the separator token.
            embedding_dim: dimension of hidden units in the sentence encoder (e.g., 768 for BERT).
        '''
        super(SSClassifier_Cohan, self).__init__()
        self.bert = transformers.AutoModel.from_pretrained(encoder_id)
        self.SEP_id = sep_id
        dropout = torch.nn.Dropout(dropout_rate)
        n_classes = n_classes
        
        dense_out = torch.nn.Linear(embedding_dim, n_classes)
        torch.nn.init.xavier_uniform_(dense_out.weight)
        self.classifier = torch.nn.Sequential(
            dropout, dense_out
        )

    def forward(self, input_ids, attention_mask):
        '''
        Each call to this method process a batch of sentence chunks. The model takes account of 
        all sentences in chunk, but not all sentences in a document. There is no data sharing 
        among the chunks, so the model is unware of when chunks are from same document.
        A chunk aggregates many sentences which are separated by 
        a separator token. The first token in the block must be the classification token (e.g., [CLS] for BERT).
        This method returns one logit tensor for each sentence in the batch.
        Arguments:
            input_ids : tensor of shape (batch_size, n of tokens in block)
            attention_mask : tensor of shape (batch_size, n of tokens in block)
        Returns:
            logits : tensor of shape (n of sentences in batch, n of classes)
        '''
        output_1 = self.bert(
            input_ids=input_ids,              # input_ids.shape: (batch_size, seq_len)
            attention_mask=attention_mask     # attention_mask.shape: (batch_size, seq_len)
        )
        
        # yelds a logit tensor for each SEP token
        # as the blocks may have different number of sentences, we have to iterate blocks
        embeddings = None
        for i in range(input_ids.shape[0]):
            idx_sep = torch.nonzero(input_ids[i] == self.SEP_id, as_tuple=True)[0]
            for idx in idx_sep: # iterates SEP tokens in current block
                sep_emb = output_1.last_hidden_state[i, idx, :] # gets embeddings of a SEP token
                if embeddings is None:
                    embeddings = sep_emb
                else:
                    embeddings = torch.vstack((embeddings, sep_emb))
        # embeddings.shape: (number of sentences in batch, hidden dimension)
        logits = self.classifier(embeddings) # logits.shape: (number of sentences in batch, num of classes)
        
        return logits

class SSClassifier_Cohan_Longformer(torch.nn.Module):
    def __init__(self, encoder_id, n_classes, dropout_rate, sep_id, embedding_dim):
        '''
        Creates a classifier of sequential sentences similar to the one proposed in the Pretrained Language Models 
        for Sequential Sentence Classification paper (Cohan et al, 2019). This model comprises a pre-trained 
        Longformer model as encoder and a classification head. The last is a single fully-connected layer.
        Arguments:
            encoder_id: ID (string) of the encoder model in Hugging Faces repository.
            n_classes: number of classes.
            dropout_rate: dropout rate of classification layers.
            sep_id: the ID of the </s> token.
            embedding_dim: dimension of hidden units in the sentence encoder (e.g., 768 for longformer-base-4096).
        '''
        super(SSClassifier_Cohan_Longformer, self).__init__()
        self.encoder = transformers.AutoModel.from_pretrained(encoder_id)
        self.SEP_id = sep_id
        dropout = torch.nn.Dropout(dropout_rate)
        n_classes = n_classes
        
        dense_out = torch.nn.Linear(embedding_dim, n_classes)
        torch.nn.init.xavier_uniform_(dense_out.weight)
        self.classifier = torch.nn.Sequential(
            dropout, dense_out
        )

    def forward(self, input_ids, attention_mask):
        '''
        Each call to this method process a batch of sentence blocks. The model takes account of 
        all sentences in block, but not all sentences in a document. There is no data sharing 
        among the blocks, so the model is unware of when blocks are from same document.
        A block of sentences aggregates many sentences which are separated by 
        a </s> token. The first token in the block must be the <s> token.
        This method returns one logit tensor for each sentence in the batch.
        Arguments:
            input_ids : tensor of shape (batch_size, n of tokens in block)
            attention_mask : tensor of shape (batch_size, n of tokens in block)
            global_attention_mask : tensor of shape (batch_size, n of tokens in block)
        Returns:
            logits : tensor of shape (n of sentences in batch, n of classes)
        '''
        # The global_attention_mask tells which tokens must have global attention. 
        # Here, <s> and </s> have global attention
        global_attention_mask = torch.zeros(
            input_ids.shape, 
            dtype=torch.long, 
            device=input_ids.device
        )
        global_attention_mask[:, 0] = 1 # global attention for <s> since it must be the first token in a block
        idx_sep = torch.nonzero(input_ids == self.SEP_id, as_tuple=True)
        for i in range(idx_sep[0].shape[0]):
            global_attention_mask[idx_sep[0][i], idx_sep[1][i]] = 1 # global attention for </s>
        
        output_1 = self.encoder(
            input_ids=input_ids,                         # input_ids.shape: (batch_size, seq_len)
            attention_mask=attention_mask,               # attention_mask.shape: (batch_size, seq_len)
            global_attention_mask=global_attention_mask  # global_attention_mask.shape: (batch_size, seq_len)
        )
        
        # yelds a logit tensor for each </s> token
        # as the blocks may have different number of sentences, we have to iterate blocks
        embeddings = None
        for i in range(input_ids.shape[0]):
            idx_sep = torch.nonzero(input_ids[i] == self.SEP_id, as_tuple=True)[0]
            for idx in idx_sep: # iterates </s> tokens in current block
                sep_emb = output_1.last_hidden_state[i, idx, :] # gets embeddings of a </s> token
                if embeddings is None:
                    embeddings = sep_emb
                else:
                    embeddings = torch.vstack((embeddings, sep_emb))
        # embeddings.shape: (number of sentences in batch, hidden dimension)
        logits = self.classifier(embeddings) # logits.shape: (number of sentences in batch, num of classes)
        
        return logits
    
def enforce_max_sent_per_chunk(sentences, labels, max_sent_per_chunk):
    """
    Splits a document with goal to produce splits that are of almost
    equal size to avoid the scenario where all splits are of size
    max_n_sentences then the last split is 1 or 2 sentences. 
    This would result into losing context around the edges of each examples.
    Code adapted from https://github.com/allenai/sequential_sentence_classification/blob/master/sequential_sentence_classification/dataset_reader.py
    Arguments:
        sentences (list of string): the sequence of sentences in a document.
        labels (list of string): the labels of the sentences.
        max_sent_per_chunk (integer): the maximum number of sentences in a chunk 
            without taking the overlaped sentences in account.
    Returns:
        List of list of string. Each sublist represents a chunk and its elements are the 
        chunk's sentences.
        List of list of string. Each sublist represents a chunk and its elements are the 
        labels of the sentences in the chunk.
    """
    assert len(sentences) == len(labels)

    if len(sentences) > max_sent_per_chunk:
        i = len(sentences) // 2
        s1, l1 = enforce_max_sent_per_chunk(sentences[:i], labels[:i], max_sent_per_chunk)
        s2, l2 = enforce_max_sent_per_chunk(sentences[i:], labels[i:], max_sent_per_chunk)
        return s1 + s2, l1 + l2
    else:
        return [sentences], [labels]

class CohanDataset(Dataset):
    def __init__(self, docs_dic, labels_to_idx, tokenizer, max_sent_len, max_sent_per_block, max_block_len):
        """
        Dataset of inputs for a SSC model. Each item in the dataset represents a block of sentences, i.e., 
        several sentences separated by a [SEP] token. Each block is padded with zeros untill it reaches the 
        max_block_len length. Sentences longer than max_sent_len tokens are truncated.
        Arguments:
            docs_dic: a dictionary whose each key is a document id (string) and each value is a pandas DataFrame
                      containing sentence and label columns.
            labels_to_idx: 
            tokenizer: tokenizer of the exploited model.
            max_sent_len: maximum number of tokens in a sentence.
            max_sent_per_block: maximum number of sentences in a block.
            max_block_len: maximum number of tokens in a block. It's usually defined by the exploited model 
                        (e.g., 512 for BERT).
        """
        self.n_documents = len(docs_dic)
        # Be aware as blocks may have different numbers of sentences, they may have different numbers of labels.
        # For each block, we have number of labels equals to number of sentences.
        blocks = []      # list of list of strings (one list of sentences for each block). shape: (n blocks, n sentences in block)
        self.labels = [] # list of list of strings (one list of labels for each block). shape: (n blocks, n labels in block)
        for doc_id, doc_df in docs_dic.items(): # iterates documents
            blks, lbls = enforce_max_sent_per_chunk(
                doc_df['sentence'].to_list(), 
                doc_df['label'].to_list(), 
                max_sent_per_block
            )
            blocks.extend(blks)
            self.labels.extend(lbls)
        
        # self.labels: list of list of strings (one list of labels for each block). shape: (n blocks, n labels in block)
        for b, l in zip(blocks, self.labels):
            if len(b) != len(l):
                print('ERROR in the block:', b)
                raise ValueError(f'Number of sentences different of the number of labels: {len(b)} != {len(l)}')
        
        self.targets = []  # list of tensors (one 1-D tensor for each block). shape: (n blocks, n labels in block)
        for label_list in self.labels:
            self.targets.append(
                torch.tensor([labels_to_idx[l] for l in label_list], dtype=torch.long)
            )
        
        # adjusting the length of each sentence
        for block in blocks:
            for i in range(len(block)):
                sent_ids = tokenizer.encode(
                    block[i], 
                    add_special_tokens=False, 
                    padding=False,
                    truncation=True,
                    max_length=max_sent_len
                )
                block[i] = tokenizer.decode(sent_ids)
        # tokeninzing blocks
        self.blocks = []
        for block in blocks:
            self.blocks.append(f' {tokenizer.sep_token} '.join(block))
        tokens = tokenizer(
            self.blocks, 
            add_special_tokens=True, 
            return_token_type_ids=False, 
            return_attention_mask=True, 
            truncation=True, 
            padding='max_length',
            max_length=max_block_len,
            return_tensors='pt'
        )
        self.input_ids = tokens['input_ids']             # tensor for all blocks. shape: (n blocks, max_block_len)
        self.attention_masks = tokens['attention_mask']  # tensor for all blocks. shape: (n blocks, max_block_len)
        
        # checking whether the number of SEP tokens is equal to then number of labels as 
        # truncation might delete SEP tokens
        SEP_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
        n_sep_tokens = torch.count_nonzero(self.input_ids == SEP_id)
        n_labels = sum(map(lambda l: len(l), self.labels))
        if n_sep_tokens != n_labels:
            raise ValueError((
                f'Number of SEP tokens ({n_sep_tokens}) different from the number of labels ({n_labels}). '
                'Decrease max_sent_len, or increase max_block_len, or decrease max_sent_per_block to avoid this.'
            ))

    def __getitem__(self, index):
        return {
            'ids': self.input_ids[index],        # tensor of shape (max_tokens)
            'mask': self.attention_masks[index], # tensor of shape (max_tokens)
            'targets': self.targets[index],      # tensor of shape (n of labels in the block)
            'labels': self.labels[index]         # list of size (n of labels in the block)
        }
    
    def __len__(self):
        return len(self.labels)
    
def collate_batch_cohan(batch):
    '''
    Prepares a batch of dataset items.
    Arguments:
        batch: list of dataset items (dictionaries).
    Returns:
        A dictionary with following items:
            'ids': tensor of input ids. Shape: (n blocks in batch, max_tokens)
            'maks': tensor of attention masks. Shape: (n blocks in batch, max_tokens)
            'targets': tensor of golden class ids. Shape: (n sentences)
            'labels': list of golden labels. Shape: (n sentences)
    '''
    labels = []
    targets = []
    input_ids = None
    masks = None
    for entry in batch:
        labels.extend(entry['labels'])
        targets.extend(entry['targets'])
        if input_ids is None:
            input_ids = entry['ids'].reshape(1,-1) # reshape to assure a 2-D tensor when the batch contains just one element
            masks = entry['mask'].reshape(1,-1)
        else:
            input_ids = torch.vstack((input_ids, entry['ids']))
            masks = torch.vstack((masks, entry['mask']))
    return {
        'ids': input_ids, 
        'mask': masks, 
        'targets': torch.tensor(targets, dtype=torch.long), 
        'labels': labels
    }

# **************** Overlaped chunks ****************
    
def doc_to_overlapping_chunks(sentences, labels, window_size, max_sent_per_chunk):
    """
    Splits a document in chuncks of sentences. There is overlapping 
    between following chunks.
    Arguments:
        sentences (list of string): the sequence of sentences in a document.
        labels (list of string): the labels of the sentences.
        window_size (integer): the number of overlapping sentences between chunks. The code 
            will fail if the window size is larger than the neighbours chunks.
        max_sent_per_chunk (integer): the maximum number of sentences in a chunk 
            without taking the overlaped sentences in account.
    Returns:
        chunks of sentences: list of list of string.
        chunks of labels: list of list of string.
        left_overlapings: list of list of string. Each sublist has window_size elements, 
            excepts for the first one which is an empty list.
        right_overlapings: list of list of string. Each sublist has window_size elements, 
            excepts for the last one which is an empty list.
    """
    chunks_s, chunks_l = enforce_max_sent_per_chunk(sentences, labels, max_sent_per_chunk)
    left_overlapings = []
    right_overlapings = []
    for i in range(len(chunks_s)):
        # left side
        if i == 0:
            t = []
        else:
            t = chunks_s[i-1][-window_size:]
        left_overlapings.append(t)
        # right side
        if i == len(chunks_s) - 1:
            t = []
        else:
            t = chunks_s[i+1][:window_size]
        right_overlapings.append(t)
        
    return chunks_s, chunks_l, left_overlapings, right_overlapings

def merge_chunks_and_overlaps(chunks, left_overlaps, right_overlaps):
    """
    Merges chunks and thier respective overlapped sentences into a unique data structure.
    Arguments:
        chunks (list of list of string): each item represents a chunk.
        left_overlaps (list of list of string): each item is the list of left overlaped sentences of the respective chunk.
        right_overlaps (list of list of string): each item is the list of right overlaped sentences of the respective chunk.
    Returns:
        overlapped_chunks (list of list of string): each item represents a chunk merged/overlaped with its neighbor sentences
        count_overlaped_sentences (list of tuples): each tuple stores the number of overlaped setences on the left and on the right for the respective chunk
    """
    overlapped_chunks = []
    count_overlaped_sentences = []
    for l, c, r in zip(left_overlaps, chunks, right_overlaps):
        overlapped_chunks.append(l + c + r)
        count_overlaped_sentences.append((len(l), len(r))) # each tuple stores the number of overlaped setences on the left and on the right for the respective chunk
    return overlapped_chunks, count_overlaped_sentences

class OverlapedChunksDataset(Dataset):
    def __init__(self, docs_dic, tokenizer, max_sent_len, max_sent_per_chunk, max_chunk_len, window_len, labels_to_targets):
        """
        Dataset of inputs for a SSC model. Each item in the dataset represents a chunk of sentences.
        There are overlaping of sentences between following chunks. Sentences are separated by a separator token 
        defined by the tokenizer. Each chunk is padded with zeros untill it reaches the 
        max_chunk_len length. Sentences longer than max_sent_len tokens are truncated.
        Arguments:
            docs_dic: a dictionary whose each key is a document id (string) and each value is a pandas DataFrame
                      containing sentence and label columns.
            tokenizer: tokenizer of the exploited model.
            max_sent_len (integer): maximum number of tokens in a sentence.
            max_sent_per_chunk (integer): maximum number of sentences in a chunk without taking the overlaped sentences in account.
            max_chunk_len (integer): maximum number of tokens in a chunk. It's usually defined by the exploited model 
                        (e.g., 512 for BERT).
            window_len (integer): number of overlapping sentences in each side of a chunk.
            labels_to_targets (dictionary - string / int): maps each label to a integer target.
        """
        self.n_documents = len(docs_dic)
        self.window_len = window_len
        # Be aware as chunks may have different numbers of sentences, they may have different numbers of labels.
        # For each chunk, we have number of labels equals to number of sentences.
        chunks = []             # list of list of strings (one list of sentences for each chunk). shape: (n chunks, n sentences in chunk)
        self.labels = []        # list of list of strings (one list of labels for each chunk). shape: (n chunks, n labels in chunk)
        self.n_overlaped_sentences = [] # list of tuples (one tuple per chunk). List shape: (n chunks). Tuple shape: (2).
        for doc_id, doc_df in docs_dic.items(): # iterates documents:
            # get chunks
            blks, lbls, left_sents, right_sents = doc_to_overlapping_chunks(
                doc_df['sentence'].to_list(), 
                doc_df['label'].to_list(), 
                window_len, 
                max_sent_per_chunk, 
            )
            # overlaps chunks with their respective neighbour sentences
            overlapped_chunks, overlaping_counts = merge_chunks_and_overlaps(blks, left_sents, right_sents)
            chunks.extend(overlapped_chunks)
            self.labels.extend(lbls)
            self.n_overlaped_sentences.extend(overlaping_counts)
        
        # checking the chunks' lengths
        for ch, lb, (left_n, right_n) in zip(chunks, self.labels, self.n_overlaped_sentences):
            n_sentences_in_chunk = len(ch) - left_n - right_n
            if n_sentences_in_chunk != len(lb):
                print('ERROR in the chunk:', ch)
                raise ValueError(f'Number of core sentences is different from the number of labels: {n_sentences_in_chunk} != {len(lb)}')
        
        # generating targets
        self.targets = []  # list of tensors (one 1-D tensor for each chunk). shape: (n chunks, n labels in chunk)
        for label_list in self.labels:
            self.targets.append(
                torch.tensor([labels_to_targets[l] for l in label_list], dtype=torch.long)
            )
        
        # adjusting the length of each sentence
        for chunk in chunks:
            for i in range(len(chunk)):
                sent_ids = tokenizer.encode(
                    chunk[i], 
                    add_special_tokens=False, 
                    padding=False,
                    truncation=True,
                    max_length=max_sent_len
                )
                chunk[i] = tokenizer.decode(sent_ids)
        
        # tokenizing chunks
        self.chunks = []
        for chunk in chunks:
            self.chunks.append(f' {tokenizer.sep_token} '.join(chunk))
        tokens = tokenizer(
            self.chunks, 
            add_special_tokens=True, 
            return_token_type_ids=False, 
            return_attention_mask=True, 
            truncation=True, 
            padding='max_length',
            max_length=max_chunk_len,
            return_tensors='pt'
        )
        self.input_ids = tokens['input_ids']             # tensor for all chunks. shape: (n chunks, max_chunk_len)
        self.attention_masks = tokens['attention_mask']  # tensor for all chunks. shape: (n chunks, max_chunk_len)
        
        # checking whether the number of separator tokens is the expected one since 
        # truncation might delete separator tokens
        SEP_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
        n_sep_tokens = torch.count_nonzero(self.input_ids == SEP_id)
        n_labels = sum(map(lambda l: len(l), self.labels))
        n_sent_at_left = sum(map(lambda l: l[0], self.n_overlaped_sentences))
        n_sent_at_right = sum(map(lambda l: l[1], self.n_overlaped_sentences))
        n_expected = n_labels + n_sent_at_left + n_sent_at_right
        if n_sep_tokens != n_expected:
            raise ValueError((
                f'Number of separator tokens ({n_sep_tokens}) different from the expected one ({n_expected}). '
                'Decrease max_sent_len or max_sent_per_chunk, or increase max_chunk_len to avoid this.'
            ))
    
    def __getitem__(self, index):
        return {
            'ids': self.input_ids[index],        # tensor of shape (max_tokens)
            'mask': self.attention_masks[index], # tensor of shape (max_tokens)
            'targets': self.targets[index],      # tensor of shape (n of labels in the chunk)
            'labels': self.labels[index],        # list of size (n of labels in the chunk)
            'n_overlaped_sentences': self.n_overlaped_sentences[index] # tuple[0]: n sentences on left, tuple[1]: n sentences on right
        }
    
    def __len__(self):
        return len(self.labels)

def collate_OverlapedChunksDataset(batch):
    '''
    Prepares a batch of dataset items. It expects to work with items from a 
    OverlapedChunksDataset.
    Arguments:
        batch: list of dataset items (dictionaries).
    Returns:
        A dictionary with following items:
            'ids': tensor of input ids. Shape: (n blocks in batch, max_tokens)
            'maks': tensor of attention masks. Shape: (n blocks in batch, max_tokens)
            'targets': tensor of golden class ids. Shape: (n sentences)
            'labels': list of golden labels. Shape: (n sentences)
            'n_overlaped_sentences': list of tuples. Shape: (n blocks in batch)
    '''
    labels = []
    targets = []
    n_overlaped_sentences = []
    input_ids = None
    masks = None
    for entry in batch:
        labels.extend(entry['labels'])
        targets.extend(entry['targets'])
        n_overlaped_sentences.append(entry['n_overlaped_sentences'])
        if input_ids is None:
            input_ids = entry['ids'].reshape(1,-1) # reshape to assure a 2-D tensor when the batch contains just one element
            masks = entry['mask'].reshape(1,-1)
        else:
            input_ids = torch.vstack((input_ids, entry['ids']))
            masks = torch.vstack((masks, entry['mask']))
    return {
        'ids': input_ids, 
        'mask': masks, 
        'targets': torch.tensor(targets, dtype=torch.long), 
        'labels': labels, 
        'n_overlaped_sentences': n_overlaped_sentences
    }

class SSClassifier_VanBerg(torch.nn.Module):
    def __init__(self, encoder_id, n_classes, dropout_rate, sep_id, embedding_dim):
        '''
        Creates a classifier of sequential sentences as proposed in the Context in Informational Bias 
        Detection paper (Van de Berg et Markert, 2020). This model comprises a pre-trained sentence 
        encoder and a classification head. The sentence encoder must be a model following BERT architecture 
        such as BERT, RoBERTa, or ALBERT. The classification is a single fully-connected layer.
        Arguments:
            encoder_id: ID (string) of the encoder model in Hugging Faces repository.
            n_classes: number of classes.
            dropout_rate: dropout rate of classification layers.
            sep_id: the ID of the separator token (e.g., [SEP] for BERT).
            embedding_dim: dimension of hidden units in the sentence encoder (e.g., 768 for BERT).
        '''
        super(SSClassifier_VanBerg, self).__init__()
        self.encoder = transformers.AutoModel.from_pretrained(encoder_id)
        self.SEP_id = sep_id
        dropout = torch.nn.Dropout(dropout_rate)
        n_classes = n_classes
        
        dense_out = torch.nn.Linear(embedding_dim, n_classes)
        torch.nn.init.xavier_uniform_(dense_out.weight)
        self.classifier = torch.nn.Sequential(
            dropout, dense_out
        )

    def forward(self, input_ids, attention_mask, overlapping_counts):
        '''
        Each call to this method process a batch of sentence chunks. A chunk has core sentences 
        and overlapping sentences, i.e., edge sentences from the neighbour chunks. The overlapping 
        sentences are a way to share context among chunks. 
        A chunk aggregates many sentences which are separated by a separator token. 
        The first token in the block must be the classification token (e.g., [CLS] for BERT).
        This method returns one logit tensor for each core sentence in the batch.
        Arguments:
            input_ids : tensor of shape (batch_size, n of tokens in chunk)
            attention_mask : tensor of shape (batch_size, n of tokens in chunk)
            overlapping_counts : list of tuples (one tuple per chunk/batch). Each tuple stores the 
                number of overlapping sentences at left and at right.
        Returns:
            logits : tensor of shape (n of sentences in batch, n of classes)
        '''
        output_1 = self.encoder(
            input_ids=input_ids,              # input_ids.shape: (batch_size, seq_len)
            attention_mask=attention_mask     # attention_mask.shape: (batch_size, seq_len)
        )
        
        # yelds a logit tensor for each SEP token that represents a core sentence
        # as the chunks may have different number of sentences, we have to iterate chunks
        embeddings = None
        for i in range(input_ids.shape[0]): # iterates batches/chunks
            idx_sep = torch.nonzero(input_ids[i] == self.SEP_id, as_tuple=True)[0] # SEP indexes from the current chunk
            for j in range(overlapping_counts[i][0], idx_sep.shape[0] - overlapping_counts[i][1]): # range is adjusted to not include SEP tokens of overlapping sentences
                sep_emb = output_1.last_hidden_state[i, j, :] # gets embeddings of a SEP token
                if embeddings is None:
                    embeddings = sep_emb
                else:
                    embeddings = torch.vstack((embeddings, sep_emb))
        # embeddings.shape: (number of core sentences in batch, hidden dimension)
        logits = self.classifier(embeddings) # logits.shape: (number of core sentences in batch, num of classes)
        
        return logits
    
class SSClassifier_VanBerg_Longformer(torch.nn.Module):
    def __init__(self, encoder_id, n_classes, dropout_rate, sep_id, embedding_dim):
        '''
        Creates a classifier of sequential sentences similar to the one proposed in the Pretrained Language Models 
        for Sequential Sentence Classification paper (Cohan et al, 2019). This model comprises a pre-trained 
        Longformer model as encoder and a classification head. The last is a single fully-connecte layer.
        Arguments:
            encoder_id: ID (string) of the encoder model in Hugging Faces repository.
            n_classes: number of classes.
            dropout_rate: dropout rate of classification layers.
            sep_id: the ID of the </s> token.
            embedding_dim: dimension of hidden units in the sentence encoder (e.g., 768 for longformer-base-4096).
        '''
        super(SSClassifier_VanBerg_Longformer, self).__init__()
        self.encoder = transformers.AutoModel.from_pretrained(encoder_id)
        self.SEP_id = sep_id
        dropout = torch.nn.Dropout(dropout_rate)
        n_classes = n_classes
        
        dense_out = torch.nn.Linear(embedding_dim, n_classes)
        torch.nn.init.xavier_uniform_(dense_out.weight)
        self.classifier = torch.nn.Sequential(
            dropout, dense_out
        )

    def forward(self, input_ids, attention_mask, overlapping_counts):
        '''
        Each call to this method process a batch of sentence chunks. A chunk has core sentences 
        and overlapping sentences, i.e., edge sentences from the neighbour chunks. The overlapping 
        sentences are a way to share context among chunks. 
        A chunk aggregates many sentences which are separated by a separator token. 
        The first token in the block must be the <s> token.
        This method returns one logit tensor for each core sentence in the batch.
        Arguments:
            input_ids : tensor of shape (batch_size, n of tokens in chunk)
            attention_mask : tensor of shape (batch_size, n of tokens in chunk)
            overlapping_counts : list of tuples (one tuple per chunk/batch). Each tuple stores the 
                number of overlapping sentences at left and at right.
        Returns:
            logits : tensor of shape (n of sentences in batch, n of classes)
        '''
        # The global_attention_mask tells which tokens must have global attention. 
        # Here, <s> and </s> have global attention
        global_attention_mask = torch.zeros(
            input_ids.shape, 
            dtype=torch.long, 
            device=input_ids.device
        )
        global_attention_mask[:, 0] = 1 # global attention for <s> since it must be the first token in a block
        idx_sep = torch.nonzero(input_ids == self.SEP_id, as_tuple=True)
        for i in range(idx_sep[0].shape[0]):
            global_attention_mask[idx_sep[0][i], idx_sep[1][i]] = 1 # global attention for </s>
        
        output_1 = self.encoder(
            input_ids=input_ids,                         # input_ids.shape: (batch_size, seq_len)
            attention_mask=attention_mask,               # attention_mask.shape: (batch_size, seq_len)
            global_attention_mask=global_attention_mask  # global_attention_mask.shape: (batch_size, seq_len)
        )
        
        # yelds a logit tensor for each </s> token that represents a core sentence
        # as the blocks may have different number of sentences, we have to iterate blocks
        embeddings = None
        for i in range(input_ids.shape[0]): # iterates batches/chunks
            idx_sep = torch.nonzero(input_ids[i] == self.SEP_id, as_tuple=True)[0] # </s> indexes from the current chunk
            for j in range(overlapping_counts[i][0], idx_sep.shape[0] - overlapping_counts[i][1]): # range is adjusted to not include </s> tokens of overlapping sentences
                sep_emb = output_1.last_hidden_state[i, j, :] # gets embeddings of a </s> token
                if embeddings is None:
                    embeddings = sep_emb
                else:
                    embeddings = torch.vstack((embeddings, sep_emb))
        # embeddings.shape: (number of sentences in batch, hidden dimension)
        logits = self.classifier(embeddings) # logits.shape: (number of core sentences in batch, num of classes)
        
        return logits