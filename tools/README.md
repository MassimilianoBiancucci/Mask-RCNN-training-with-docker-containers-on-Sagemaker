# List of useful scripts and commands

### ECR repository docker setup

This command is needed to give to docker needed permissions for pull and push repositories.

```bash
docker login -u AWS -p $(aws ecr get-login-password) 011827850615.dkr.ecr.eu-west-1.amazonaws.com
```