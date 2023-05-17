Code and artifats of the **Evaluating Recent Legal Rhetorical Role Labeling Approaches Supported by Transformer Encoders** paper.

The **reports** folder holds the reports with the results of experiments.

Files related to specific models have the model name as prefix. For example, the files `pe_app.py`, `pe_models.py`, `pe_run_InCaseLaw.py` and `pe_run_RoBERTa.py` concern the PE-S and PE-S models. The prefix `mixup` relates the Mixup-A models while `mixup2` related the Mixup-B models.

### Running models

There is a running script for each model. For example, to run the **DFCSC-CLS RoBERTa** model we execute the `dfcsc_cls_run_RoBERTa.py` file. Running a model yields the respective report file. There is no command line parameters.

This repository does not contain the original dataset, though it is available at this [link](https://github.com/Exploration-Lab/Rhetorical-Roles). The original dataset path is set in the `data_manager.py` file. Before running a model, set this path accordingly the location of the dataset on your system.

The hyperparameters of a model can be set in the respective run script. In the following we describe such hyperparameters.

Generic hyperparameters (i.e., they are available in all models):
- `ENCODER_ID`: identifier of the exploited pre-trained Transformer model from the Hugging Face repository (https://huggingface.co/models).
- `MODEL_REFERENCE`: name utilized to reference the model in the reports.
- `MAX_SEQUENCE_LENGTH`: number of tokens in a chunk or sequence. It is usually set equals to the maximum sequence supported by the pre-trained model.
- `EMBEDDING_DIM`: the embedding dimension of a token embedding. It is determined by the choosen pre-trained model.
- `N_EPOCHS`: the number of fine-tuning epochs.
- `LEARNING_RATE`: the initial learning rate of the fine-tuning procedure.
- `BATCH_SIZE`: batch size of the fine-tuning procedure.
- `DROPOUT_RATE`: dropout rate of the fine-tuning procedure.
- `USE_MLP`: boolean value to indicate if the classifier must comprise one (`False`) or two (`True`) dense layers.
- `n_iterations`: number of executions of the model. Each execution adopts a different random number seed value.
- `weight_decay`: weight decay value of the Adam optimizer.
- `eps`: epsilon value of the Adam optimizer.
- `use_mock`: boolean value to indicate if it should to use a mock model instead a real one. This is used as way to speed the runing time when the code is being validated.
- `n_documents`: number of documents to be used to train and evaluate a model. This is used as way to speed the runing time when the code is being validated.
- `SAVE_BEST`: if `True`, the weights of the best epoch are saved in disk. If `False`, nothing is saved.

SingleSC models:
- `freeze_layers`: If `True`, half of the first transformer layers are freezed and thus they are not updated during fine-tuning. If `False`, all the transformer layers are updated.
- `warmup`: indicates if a learning rate warmup procedure must be adopted (`True`) or not (`False`).

DFCSC-CLS and DFCSC-SEP models:
- `MIN_CONTEXT_LENGTH` (*m*): the desired minimum number of tokens in the edges of a chunk.

Cohan models:
- `MAX_SENTENCE_LENGTH`: maximum number of tokens in a core sentence.
- `MAX_SENTENCES_PER_BLOCK`: maximum number of core sentences in a chunk.
- `CHUNK_LAYOUT`: the layout of the chunk. Set `Cohan` to indicate a chunk without shared edges.

Sharing edges models:
- `MAX_SENTENCE_LENGTH`: maximum number of tokens in a core sentence.
- `MAX_SENTENCES_PER_BLOCK`: maximum number of core sentences in a chunk.
- `WINDOW_LENGTH`: maximum number of shared sentences in each edge of a chunk.
- `CHUNK_LAYOUT`: the layout of the chunk. Set `VanBerg` to indicate a chunk with shared edges.
- `MAX_SEQUENCE_LENGTH`: number of tokens in a chunk (*c_len*). The provided value is adjusted in order to minimize the use of `PAD` tokens and so the actual value may be lower than the provided one.




