import cohan_app

ENCODER_ID = 'law-ai/InCaseLawBERT' # id from HuggingFace
MODEL_REFERENCE = 'InCaseLaw'
MAX_SEQUENCE_LENGTH = 512   # max number of tokens in a chunk
EMBEDDING_DIM = 768
BATCH_SIZE = 16
DROPOUT_RATE = 0.2

DATASET = '7_roles' # '7_roles' or '4_roles'
MAX_SENTENCE_LENGTH = 85
MAX_SENTENCES_PER_BLOCK = 7
CHUNK_LAYOUT = 'Cohan'

N_EPOCHS = 4
LEARNING_RATE = 1e-5

train_params = {}
train_params['encoder_id'] = ENCODER_ID
train_params['model_reference'] = MODEL_REFERENCE
train_params['dataset'] = DATASET
train_params['max_seq_len'] = MAX_SEQUENCE_LENGTH
train_params['embedding_dim'] = EMBEDDING_DIM
train_params['batch_size'] = BATCH_SIZE
train_params['dropout_rate'] = DROPOUT_RATE
train_params['n_epochs'] = N_EPOCHS
train_params['learning_rate'] = LEARNING_RATE
train_params['chunk_layout'] = CHUNK_LAYOUT
train_params['max_sent_len'] = MAX_SENTENCE_LENGTH
train_params['max_sent_per_block'] = MAX_SENTENCES_PER_BLOCK

train_params['weight_decay'] = 1e-3
train_params['eps'] = 1e-8

#train_params['n_documents'] = 1
train_params['n_iterations'] = 5
train_params['use_mock'] = False

cohan_app.evaluate_model(train_params)
