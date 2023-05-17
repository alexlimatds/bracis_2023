import dfcsc_sep_app

ENCODER_ID = 'roberta-base' # id from HuggingFace
MODEL_REFERENCE = 'RoBERTa'
MAX_SEQUENCE_LENGTH = 512   # max number of tokens in a chunk
EMBEDDING_DIM = 768
BATCH_SIZE = 16
DROPOUT_RATE = 0.2

DATASET = '4_roles' # '7_roles' or '4_roles'
MIN_CONTEXT_LENGTH = 250

N_EPOCHS = 4
LEARNING_RATE = 1e-5

train_params = {}
train_params['encoder_id'] = ENCODER_ID
train_params['model_reference'] = MODEL_REFERENCE
train_params['dataset'] = DATASET
train_params['max_seq_len'] = MAX_SEQUENCE_LENGTH
train_params['min_context_len'] = MIN_CONTEXT_LENGTH
train_params['embedding_dim'] = EMBEDDING_DIM
train_params['batch_size'] = BATCH_SIZE
train_params['dropout_rate'] = DROPOUT_RATE
train_params['n_epochs'] = N_EPOCHS
train_params['learning_rate'] = LEARNING_RATE

train_params['weight_decay'] = 1e-3
train_params['eps'] = 1e-8

#train_params['n_documents'] = 1
train_params['n_iterations'] = 5
train_params['use_mock'] = False

dfcsc_sep_app.evaluate_model(train_params)
