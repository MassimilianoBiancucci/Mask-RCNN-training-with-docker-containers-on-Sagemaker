# Using the SageMaker Python SDK

Per lanciare il container da un sagemaker notebook si deve usare:

```python
from sagemaker.estimator import Estimator

estimator = Estimator([...])
```

la sinopsi e':

```python
class sagemaker.estimator.Estimator(image_uri, role, instance_count=None, instance_type=None, volume_size=30, volume_kms_key=None, max_run=86400, input_mode='File', output_path=None, output_kms_key=None, base_job_name=None, sagemaker_session=None, hyperparameters=None, tags=None, subnets=None, security_group_ids=None, model_uri=None, model_channel_name='model', metric_definitions=None, encrypt_inter_container_traffic=False, use_spot_instances=False, max_wait=None, checkpoint_s3_uri=None, checkpoint_local_path=None, enable_network_isolation=False, rules=None, debugger_hook_config=None, tensorboard_output_config=None, enable_sagemaker_metrics=None, profiler_config=None, disable_profiler=False, environment=None, max_retry_attempts=None, **kwargs)
```

i parametri sono:

- image_uri (str) – The container image to use for training for. </br> -> va preso da ECR.

- role (str) – An AWS IAM role (either name or full ARN). The Amazon SageMaker training jobs and APIs that create Amazon SageMaker endpoints use this role to access training data and model artifacts. After the endpoint is created, the inference code might use the IAM role, if it needs to access an AWS resource </br> -> dalla guida "Adapting Your Own Training Container" usa:

```python
from sagemaker import get_execution_role

estimator = Estimator(...
                    role=get_execution_role(),
                    ...)
```

- instance_count (int) – Number of Amazon EC2 instances to use for training.

- instance_type (str) – Type of EC2 instance to use for training, for example, ‘ml.c4.xlarge’.

- volume_size (int) – Size in GB of the EBS volume to use for storing input data during training (default: 30). Must be large enough to store training data if File Mode is used (which is the default).

- max_run (int) – Timeout in seconds for training (default: 24 * 60 * 60). After this amount of time Amazon SageMaker terminates the job regardless of its current status.

- input_mode (str) – The input mode that the algorithm supports (default: ‘File’). Valid modes:

  - ’File’ - Amazon SageMaker copies the training dataset from the S3 location to a local directory.

  - ’Pipe’ - Amazon SageMaker streams data directly from S3 to the container via a Unix-named pipe.

  This argument can be overriden on a per-channel basis using: `sagemaker.inputs.TrainingInput.input_mode` </br>
  -> negli esempi abbiamo vista usata solo la modalita' file.

- output_path (str) – S3 location for saving the training result (model artifacts and output files). If not specified, results are stored to a default bucket. If the bucket with the specific name does not exist, the estimator creates the bucket during the fit() method execution.

- base_job_name (str) – Prefix for training job name when the fit() method launches. If not specified, the estimator generates a default job name, based on the training image name and current timestamp.

- sagemaker_session (sagemaker.session.Session) – Session object which manages interactions with Amazon SageMaker APIs and any other AWS services needed. If not specified, the estimator creates one using the default AWS configuration chain.

- hyperparameters (dict) – Dictionary containing the hyperparameters to initialize this estimator with. </br> -> su "Sagemaker practical reference" fa l'encoding degli iperparametri come json, mentre qua li vuole di tipo dict. **perche'?**

