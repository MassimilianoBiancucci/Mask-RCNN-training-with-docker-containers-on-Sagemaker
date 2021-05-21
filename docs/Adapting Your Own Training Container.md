# [Adapting Your Own Training Container](https://docs.aws.amazon.com/sagemaker/latest/dg/adapt-training-container.html)

Create Dockerfile and Python training scripts

il `Dockerfile` deve avere la struttura:

```Dockerfile
# Install sagemaker-training toolkit that contains the common functionality necessary to create a container compatible with SageMaker and the Python SDK.
RUN pip3 install sagemaker-training

# Copies the training code inside the container
COPY train.py /opt/ml/code/train.py

# Defines train.py as script entrypoint
ENV SAGEMAKER_PROGRAM train.py
```

- `RUN pip install sagemaker-training` – Installs SageMaker Training Toolkit that contains the common functionality necessary to create a container compatible with SageMaker.

- `COPY train.py /opt/ml/code/train.py` – Copies the script to the location inside the container that is expected by SageMaker. The script must be located in this folder.

- `ENV SAGEMAKER_PROGRAM train.py` – Takes your training script train.py as the entrypoint script copied in the `/opt/ml/code` folder of the container. This is the only environmental variable that you must specify when you build your own container.

Per lanciare il docker:

```python
import sagemaker
from sagemaker import get_execution_role
from sagemaker.estimator import Estimator

estimator = Estimator(image_uri=image_uri --> da ECR,
                    role=get_execution_role(),
                    base_job_name='tf-custom-container-test-job',
                    instance_count=1,
                    instance_type='ml.p2.xlarge')

# start training
estimator.fit()
```
