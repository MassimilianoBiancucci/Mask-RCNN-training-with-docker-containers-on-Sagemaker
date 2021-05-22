# Use Checkpoints in Amazon SageMaker

For custom training containers and other frameworks

If you are using your own training containers, training scripts, or other frameworks not listed in the previous section, you must properly set up your training script using callbacks or training APIs to save checkpoints to the local path ('/opt/ml/checkpoints') and load from the local path in your training script. SageMaker estimators can sync up with the local path and save the checkpoints to Amazon S3.

```python
estimator = Estimator(
        ...
        checkpoint_s3_uri=checkpoint_s3_bucket -> s3://....,
        checkpoint_local_path=checkpoint_local_path -> "/opt/ml/checkpoints"
)
```

<span style="color:red">il checkpoint_local_path puo' essere cambiato con il path che gia' usa la rete in questo momento?</span>
