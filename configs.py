def config_DHMM():
    conf = {

# Model Arguments
    'input_dim': 2,
    'z_dim':16,
    'emission_dim':16,
    'trans_dim':20,
    'rnn_dim':30,

    'temp':1.0, # softmax temperature (lower --> more discrete)
    'dropout':0.5, # dropout applied to layers (0 = no dropout)

# Training Arguments
    'batch_size':20,
    'epochs':1000, # maximum number of epochs
    'min_epochs':2, # minimum number of epochs to train for

    'lr':3e-4, # autoencoder learning rate
    'beta1':0.96, # beta1 for adam
    'beta2':0.999,
    'clip_norm':20.0,  # gradient clipping, max norm       
    'weight_decay':2.0,
    'anneal_epochs':1000,
    'min_anneal':0.1,
    }
    return conf 
