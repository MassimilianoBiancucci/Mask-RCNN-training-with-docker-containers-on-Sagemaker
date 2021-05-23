import os
import argparse
import time
from directory_tree import display_tree

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # sagemaker-containers passes hyperparameters as arguments
    parser.add_argument("--hp1", type=str, default="ciao")
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

    #aspetto 10 secondi
    time.sleep(10)

    #reading the folder structure and write the result on /opt/ml/output
    tree = display_tree('/opt/ml', string_rep = True)

    print(tree)
    
    f = open("/opt/ml/output/data/tree_result.txt", "w")
    f.write(tree)
    f.close()

    for i in range(30):
            
        f = open(f"/opt/ml/output/tensorboard/record_test_{i}.txt", "w")
        f.write("test " * 100)
        f.close()

        print(f"new tensorboard record saved {i+1}/10")

        f = open(f"/opt/ml/checkpoints/checkpoint_test_{i}.txt", "w")
        f.write("prova " * 100)
        f.close()

        print(f"new checkpoint saved {i+1}/10")

        time.sleep(20)

        #reading the folder structure and write the result on /opt/ml/output
    tree = display_tree('/opt/ml', string_rep = True)

    print(tree)



