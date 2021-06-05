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

- image_uri (str) – The container image to use for training for.
</br> -> va preso da ECR.

- role (str) – An AWS IAM role (either name or full ARN). The Amazon SageMaker training jobs and APIs that create Amazon SageMaker endpoints use this role to access training data and model artifacts. After the endpoint is created, the inference code might use the IAM role, if it needs to access an AWS resource
</br> -> dalla guida "Adapting Your Own Training Container" usa:

    ```python
    from sagemaker import get_execution_role

    estimator = Estimator(...
                        role=get_execution_role(),
                        ...)
    ```

- instance_count (int) – Number of Amazon EC2 instances to use for training.

- instance_type (str) – Type of EC2 instance to use for training, for example, `ml.c4.xlarge`.

- volume_size (int) – Size in GB of the EBS volume to use for storing input data during training (default: 30). Must be large enough to store training data if File Mode is used (which is the default).

- max_run (int) – Timeout in seconds for training (default: 24 * 60 * 60). After this amount of time Amazon SageMaker terminates the job regardless of its current status.

- input_mode (str) – The input mode that the algorithm supports (default: ‘File’). Valid modes:

  - ’File’ - Amazon SageMaker copies the training dataset from the S3 location to a local directory.

  - ’Pipe’ - Amazon SageMaker streams data directly from S3 to the container via a Unix-named pipe.

  This argument can be overriden on a per-channel basis using: `sagemaker.inputs.TrainingInput.input_mode`
  </br> -> negli esempi abbiamo vista usata solo la modalita' file.

- output_path (str) – S3 location for saving the training result (model artifacts and output files). If not specified, results are stored to a default bucket. If the bucket with the specific name does not exist, the estimator creates the bucket during the fit() method execution.

- base_job_name (str) – Prefix for training job name when the fit() method launches. If not specified, the estimator generates a default job name, based on the training image name and current timestamp.

- sagemaker_session (sagemaker.session.Session) – Session object which manages interactions with Amazon SageMaker APIs and any other AWS services needed. If not specified, the estimator creates one using the default AWS configuration chain.

- hyperparameters (dict) – Dictionary containing the hyperparameters to initialize this estimator with.

- tags (list[dict]) – List of tags for labeling a training job. For more, see <https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html>.
</br> -> per questo progetto usiamo il TAG CER: 1

- model_uri (str) – URI where a pre-trained model is stored, either locally or in S3 (default: None). If specified, the estimator will create a channel pointing to the model so the training job can download it. This model can be a ‘model.tar.gz’ from a previous training job, or other artifacts coming from a different source. More information: <https://docs.aws.amazon.com/sagemaker/latest/dg/cdf-training.html#td-deserialization>

- model_channel_name (str) – Name of the channel where ‘model_uri’ will be downloaded (default: ‘model’).

- metric_definitions (list[dict]) – A list of dictionaries that defines the metric(s) used to evaluate the training jobs. Each dictionary contains two keys: ‘Name’ for the name of the metric, and ‘Regex’ for the regular expression used to extract the metric from the logs.
</br> -> i log vengono presi dallo stdout del docker, i regex devono essere ad [esempio](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-define-metrics.html):

    ```python
    metric_definitions=[
                            {
                                "Name": "loss",
                                "Regex": "Loss = (.*?);",
                            },
                            {
                                "Name": "ganloss",
                                "Regex": "GAN_loss=(.*?);",
                            },
                            {
                                "Name": "discloss",
                                "Regex": "disc_train_loss=(.*?);",
                            },
                            {
                                "Name": "disc-combined",
                                "Regex": "disc-combined=(.*?);",
                            },
                        ]
    ```

- use_spot_instances (bool) – Specifies whether to use SageMaker Managed Spot instances for training. If enabled then the max_wait arg should also be set. More information: <https://docs.aws.amazon.com/sagemaker/latest/dg/model-managed-spot-training.html> (default: False).

- max_wait (int) – Timeout in seconds waiting for spot training instances (default: None). After this amount of time Amazon SageMaker will stop waiting for Spot instances to become available (default: None).

- checkpoint_s3_uri (str) – The S3 URI in which to persist checkpoints that the algorithm persists (if any) during training. (default: None).

- checkpoint_local_path (str) – The local path that the algorithm writes its checkpoints to. SageMaker will persist all files under this path to checkpoint_s3_uri continually during training. On job startup the reverse happens - data from the s3 location is downloaded to this path before the algorithm is started. If the path is unset then SageMaker assumes the checkpoints will be provided under /opt/ml/checkpoints/. (default: None).

- rules - SageMaker Debugger rules

- debugger_hook_config

- tensorboard_output_config (TensorBoardOutputConfig) – Configuration for customizing debugging visualization using TensorBoard (default: None). For more information, see [Capture real time tensorboard data](https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_debugger.html#capture-real-time-tensorboard-data-from-the-debugging-hook). 
</br> -> esempio

    ```python
    from sagemaker.debugger import TensorBoardOutputConfig

    tensorboard_output_config = TensorBoardOutputConfig(
        s3_output_path = 's3://path/tensorboard/data/',                 
        container_local_output_path = '/local/path/tensorboard/data/'
    )

    estimator = Estimator(
        role=role,
        instance_count=1,
        instance_type=instance_type,
        tensorboard_output_config=tensorboard_output_config
    )
    ```

    la struttura [TensorBoardOutputConfig](https://sagemaker.readthedocs.io/en/stable/api/training/debugger.html#sagemaker.debugger.TensorBoardOutputConfig): </br>
    Create a tensor ouput configuration object for debugging visualizations on TensorBoard. Initialize the TensorBoardOutputConfig instance.

  - s3_output_path (str) – Optional. The location in Amazon S3 to store the output.

  - container_local_output_path (str) – Optional. The local path in the container.

- environment (dict[str, str]) – Environment variables to be set for use during training job (default: None)

- max_retry_attempts (int) – The number of times to move a job to the STARTING status. You can specify between 1 and 30 attempts. If the value of attempts is greater than zero, the job is retried on InternalServerFailure the same number of attempts as the value. You can cap the total duration for your job by setting max_wait and max_run (default: None)

## metodi

```python
[...estimator...].fit(inputs=None, wait=True, logs='All', job_name=None, experiment_config=None)
```

Train a model using the input training dataset.

The API calls the Amazon SageMaker CreateTrainingJob API to start model training. The API uses configuration you provided to create the estimator and the specified input training data to send the CreatingTrainingJob request to Amazon SageMaker.

This is a synchronous operation. After the model training successfully completes, you can call the deploy() method to host the model using the Amazon SageMaker hosting services.

- inputs (str or dict or sagemaker.inputs.TrainingInput) – Information about the training data. This can be one of three types:

  - `(str)` the S3 location where training data is saved, or a file:// path in local mode.

  - `(dict[str, str] or dict[str, sagemaker.inputs.TrainingInput])` If using multiple channels for training data, you can specify a dict mapping channel names to strings or TrainingInput() objects.

  - `(sagemaker.inputs.TrainingInput)` - channel configuration for S3 data sources that can provide additional information as well as the path to the training dataset. See sagemaker.inputs.TrainingInput() for full details.

  - `(sagemaker.session.FileSystemInput)` - channel configuration for a file system data source that can provide additional information as well as the path to the training dataset.

  -> nel nostro caso abbiamo usato il tipo `dict[str, str]`:

    ```python
    inputs = {"nome_modello" : "s3://path_su_s3"}
    ```

    in questo modo veniva messo nella cartella `/opt/ml/data/nome_modello/` **in contraddizione con quanto detto nella documentazione!** #TODO mettere link

- wait (bool) – Whether the call should wait until the job completes (default: True).

- logs ([str]) – A list of strings specifying which logs to print. Acceptable strings are “All”, “None”, “Training”, or “Rules”. </br> -> selezionando `All` si ha lo stdout e stderr del docker come outpu di questa cella, ma non in tempo reale.

- job_name (str) – Training job name. If not specified, the estimator generates a default job name based on the training image name and current timestamp. </br> -> deve essere univoco per ogni run.
