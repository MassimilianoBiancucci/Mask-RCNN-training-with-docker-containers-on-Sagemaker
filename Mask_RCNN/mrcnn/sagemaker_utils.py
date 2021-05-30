import os
import re
import json

############################################################
#  ENV VARIABLES
############################################################

def read_env_var(name, default_value=None):
    try:
        return os.environ[name]
    except:
        return default_value

def read_channels(channel_var='SM_CHANNELS'):
    var = read_env_var(channel_var, '[]')
    channels = json.loads(var)
    d = {}
    for c in channels:
        d[c] = read_env_var(f'SM_CHANNEL_{c.upper()}')
    return d


############################################################
#  LOADING FILES
############################################################

def last_checkpoint_path(checkpoints_dir, name):
    '''
    Fuction that return the path of the last checkpoint in the folder
    based on the iteration id
    '''
    files = [f for f in os.listdir(checkpoints_dir) if f.endswith('.' + 'h5')]
    matches = [re.match(r'mask_rcnn_' + name + r'_(\d{4})\.h5',f) for f in files]
    
    max_epoch = -1
    for m in matches:
        if m:
            file_epoch = int(m.group(1))
            if file_epoch > max_epoch:
                max_epoch = file_epoch
    
    # if no file was selected rise an error
    assert max_epoch > -1

    str_epoch = str(max_epoch)
    l = len(str_epoch)
    
    for i in range(4-l):
        str_epoch = '0'+str_epoch

    checkpoint_path = os.path.join(checkpoints_dir, f'mask_rcnn_{name}_{str_epoch}.h5')
    print(f'loading checkpoint: {checkpoint_path}')
    return checkpoint_path