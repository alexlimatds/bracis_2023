Code and artifacts of the **Evaluating Recent Legal Rhetorical Role Labeling Approaches Supported by Transformer Encoders** paper.

The `reports` folder holds the reports with the results of experiments.

Files related to specific models have the model name as prefix. For example, the files `pe_app.py`, `pe_models.py`, `pe_run_InCaseLaw.py` and `pe_run_RoBERTa.py` concern PE-S and PE-C models. The prefix `mixup` relates the Mixup-A models whereas `mixup2` relates the Mixup-B models.

### Running models

There is a running script for each model. For example, to run the DFCSC-CLS-RoBERTa model we execute the `dfcsc_cls_run_RoBERTa.py` file. Running a model yields the respective report file. There is no command line parameters.

This repository does not contain the original dataset, though it is available at this [link](https://github.com/Exploration-Lab/Rhetorical-Roles). The original dataset path is set in the `data_manager.py` file. Before running a model, set this path accordingly the location of the dataset on your system.

The hyperparameters of a model can be set in the respective run script. In the following we describe such hyperparameters.

Generic hyperparameters (i.e., they are available in all models):
- `ENCODER_ID`: identifier of the exploited pre-trained Transformer model from the Hugging Face repository (https://huggingface.co/models).
- `MODEL_REFERENCE`: name utilized to reference the model in the reports.
- `MAX_SEQUENCE_LENGTH`: number of tokens in a chunk or sentence. It is usually set equals to the maximum sequence supported by the pre-trained model. For DFCSC models, it works as the *c_len* hyperparameter.
- `EMBEDDING_DIM`: the embedding dimension of a token embedding. It is determined by the choosen pre-trained model.
- `N_EPOCHS`: the number of fine-tuning epochs.
- `LEARNING_RATE`: the initial learning rate of the fine-tuning procedure.
- `BATCH_SIZE`: batch size of the fine-tuning procedure.
- `DROPOUT_RATE`: dropout rate of the classification layer.
- `DATASET`: the ID/name of the dataset to be used. The options are `7_roles` and `4_roles`.
- `n_iterations`: number of executions of the model. Each execution adopts a different random seed value.
- `weight_decay`: weight decay value of the Adam optimizer.
- `eps`: epsilon value of the Adam optimizer.
- `use_mock`: boolean value to indicate if it should to use a mock model instead a real one. This is used as a way to speed the runing time when the code is being validated.
- `n_documents`: number of documents to be used to train and evaluate a model. This is used as a way to speed the runing time when the code is being validated.

PE models:
- `COMBINATION`: the operation to combine positional embeddings and sentence embeddings. The options are `S` for sum and `C` for concatenation.

DFCSC-CLS and DFCSC-SEP models:
- `MIN_CONTEXT_LENGTH` (*m_edges*): the desired minimum number of tokens in the edges of a chunk.

Cohan models:
- `MAX_SENTENCE_LENGTH`: maximum number of tokens in a sentence.
- `MAX_SENTENCES_PER_BLOCK`: maximum number of sentences in a chunk.
- `CHUNK_LAYOUT`: the layout of the chunk. It must be set with `Cohan`.

Mixup-A models:
- `MIXUP_ALPHA`: the alpha hyperparameter of Mixup method.
- `CLASSES_TO_AUGMENT`: when generating a Mixup vector, the training procedure chooses two source vectors. The first vector is always a random vector belonging to a class indicated in this hyperparamenter and the second vector is from a different class chosen at random.
- `AUGMENTATION_RATE`: the gamma hyperparameter described in the paper.
- `N_EPOCHS_ENCODER`: the total number or fine-tuning epochs of the encoder.
- `STOP_EPOCH_ENCODER`: the epoch in which the fine-tuning of the encoder will finish. It must be equal or lesser than `N_EPOCHS_ENCODER`. Remark that the sets `N_EPOCHS_ENCODER = 4`, `STOP_EPOCH_ENCODER=2` and `N_EPOCHS_ENCODER = 2`, `STOP_EPOCH_ENCODER=2` produce different results because of the learning rate schedule procedure.
- `LEARNING_RATE_ENCODER`: the initial learning rate used in the fine-tuning of the encoder.
- `N_EPOCHS_CLASSIFIER`: the number of training epochs of the classifier.
- `LEARNING_RATE_CLASSIFIER`: the initial learning rate used in the training of the classifier.

Mixup-B models:
- `MIXUP_ALPHA`: the alpha hyperparameter of Mixup method.
- `CLASSES_TO_AUGMENT`: when generating a Mixup vector, the training procedure chooses two source vectors. The first vector is always a random vector belonging to a class indicated in this hyperparamenter and the second vector is from a different class chosen at random.
- `AUGMENTATION_RATE`: the gamma hyperparameter described in the paper.
