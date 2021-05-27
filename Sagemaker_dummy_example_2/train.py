import os
import argparse
import time
from directory_tree import display_tree

def write_envs_to_file(var_name):
    path = '/opt/ml/output/data/out.txt'

    if not os.path.exists(path):
        with open(path, "w") as myfile:
            myfile.write("FILE CREATION\n")

    with open(path, "a+") as myfile:
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
    with open("/opt/ml/output/data/tree_result.txt", "w") as f:
        f.write(tree)

    print('work done!')
    print('goodbye!')
    print('-'*40)