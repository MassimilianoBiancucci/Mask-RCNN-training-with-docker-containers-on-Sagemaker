import os
import argparse
import time
from directory_tree import display_tree

path = '/opt/ml/output/data/'

def write_envs_to_file(var_name):
    
    file_path = path + "out.txt"

    if not os.path.exists(file_path):
        with open(file_path, "w") as myfile:
            myfile.write("FILE CREATION\n")

    with open(file_path, "a+") as myfile:
        myfile.write('-'*40 + '\n')
        try:
            var_content = os.environ[var_name]
            myfile.write(f"var_name={var_name}:\n{var_content}\n")

        except:
            myfile.write(f"var_name={var_name}: NOT_FOUND\n")

        finally:
            myfile.write('-'*40 + '\n')
        

if __name__ == "__main__":
    print('-'*40)
    print('hello!')

    host = os.environ["SM_CURRENT_HOST"]
    path += host + "_"

    print(f"current host: {host}")

    print("reading environment")

    test = os.environ["test"]

    print(f"the environment variable test has value of: {test}")    

    print('writing to file...')

    write_envs_to_file("SM_MODEL_DIR")

    write_envs_to_file("SM_CHANNELS")

    write_envs_to_file("SM_HPS")

    write_envs_to_file("SM_CURRENT_HOST")

    write_envs_to_file("SM_HOSTS")

    write_envs_to_file("SM_NUM_GPUS")

    write_envs_to_file("SM_NUM_CPUS")

    write_envs_to_file("SM_LOG_LEVEL")

    write_envs_to_file("SM_NETWORK_INTERFACE_NAME")

    write_envs_to_file("SM_USER_ARGS")

    write_envs_to_file("SM_INPUT_DIR")

    write_envs_to_file("SM_INPUT_CONFIG_DIR")

    write_envs_to_file("SM_RESOURCE_CONFIG")

    write_envs_to_file("SM_INPUT_DATA_CONFIG")

    write_envs_to_file("SM_TRAINING_ENV")

    tree = display_tree('/opt/ml', string_rep = True)
    with open(f"/opt/ml/output/data/{host}_tree_result.txt", "w") as f:
        f.write(tree)

    print('work done!')
    print('goodbye!')
    print('-'*40)