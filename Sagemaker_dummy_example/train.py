import os
import argparse
import time
from directory_tree import display_tree

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # sagemaker-containers passes hyperparameters as arguments
    parser.add_argument("--hp1", type=str, default="test_param")
    parser.add_argument("--hp2", type=int, default=50)
    parser.add_argument("--hp3", type=float, default=0.1)

    # This is a way to pass additional arguments when running as a script
    # and use sagemaker-containers defaults to set their values when not specified.
    parser.add_argument("--dataset", type=str, default=os.environ["SM_CHANNEL_DATASET"])

    args = parser.parse_args()

    print(args.hp1)
    print(args.hp2)
    print(args.hp3)
    print(args.dataset)

    #waiting 10 seconds
    time.sleep(10)

    # reading the folder structure and write the result on /opt/ml/output
    # (equivalent of tree bash command)
    tree = display_tree('/opt/ml', string_rep = True)

    # print the tree on stdout
    print(tree)
    
    # saving the tree into a txt file in the folder that will be saved to s3 at the end of the job
    f = open("/opt/ml/output/data/tree_result.txt", "w")
    f.write(tree)
    f.close()

    # simulating training
    for i in range(30):
        
        # simulating tensorboard data accumulation (this data should be visible in real-time) 
        f = open(f"/opt/ml/output/tensorboard/record_test_{i}.txt", "w")
        f.write("test " * 100)
        f.close()

        print(f"new tensorboard record saved {i+1}/10")

        #simulating the checkpoints
        f = open(f"/opt/ml/checkpoints/checkpoint_test_{i}.txt", "w")
        f.write("test " * 100)
        f.close()

        print(f"new checkpoint saved {i+1}/10")

        time.sleep(20)

    #reading the folder structure 
    tree = display_tree('/opt/ml', string_rep = True)

    #this time the result is only visible on the standard output
    print(tree)



