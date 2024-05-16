# SageMaker Distributed Data Example

Objective: Outline in a user friendly manner how one can perform DDP on SageMaker using W&B Launch.

# Setup

The following explains how to set up a W&B Launch Queue, set up and set up an agent polling on the queue.

## Creating A Queue

In the W&B App, navigate to `[wandb.ai/launch](http://wandb.ai/launch)` click Create Queue, and create a SageMaker queue with the following configuration. Ensure you use your own SageMaker execution role, and S3Output Path. For more information on SageMaker queue set up see [here](https://docs.wandb.ai/guides/launch/setup-launch-sagemaker).

Since this is a distributed example, we set the instance count to 2.

```yaml
RoleArn: <SM execution role>
ResourceConfig:
  InstanceType: ml.g4dn.xlarge
  InstanceCount: 2
  VolumeSizeInGB: 2
OutputDataConfig:
  S3OutputPath: <s3 path>
StoppingCondition:
  MaxRuntimeInSeconds: 3600
```

### Agent Set up

Launch an agent with a config as shown below. For more information about setting up

`launch-agent-config.yaml`

```yaml
environment:
  type: aws
  region: <region>
builder:
  type: docker
  destination: <ecr repo for run>
max_jobs: <n> # set this to the number of concurrent SM jobs you'd like to run
```

Starting the agent:

`wandb launch-agent -e <entity> -q <queue-name> -c launch-agent-config.yaml`

### Code Outline

Example Github repo:

[https://github.com/KyleGoyette/SageMaker-Launch-DDP](https://github.com/KyleGoyette/SageMaker-Launch-DDP)

### Dockerfile.wandb

This is the dockerfile used to build the container that is run in the SageMaker training job.

```yaml
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
COPY ./train_dis.py /
WORKDIR /
RUN pip install wandb
# Set NCCL and PyTorch environment variables
ENV NCCL_DEBUG=INFO
ENV TORCH_DISTRIBUTED_DEBUG=INFO

ENV NCCL_DEBUG_SUBSYS=ALL
ENV NCCL_SOCKET_IFNAME=eth0
ENV NCCL_IB_DISABLE=1
ENV NCCL_P2P_LEVEL=NVL
```

### launch-config.json

This file is used to control the values of the hyperparameters passed to the job being enqueued. Modify the values in the run_config to change their values when the code runs.

### train_dis.py

Sample script for NCCL driven Pytorch DDP training.

# Running the code

## Have the Launch Agent Build the Image

With your launch agent up and running, polling on a queue. You can now submit jobs to the queue using W&B Launch. This can be done either through git or local paths:

Local path:

`wandb launch -u /path/to/root/of/code --entry-point "python train_dis.py" -e <entity> -p <project> -q <queue-name> -c launch_config.json`

Git:

`wandb launch -u <git-uri> -g <branch-name> --entry-point "python train_dis.py" -e <entity> -p <project> -q <queue-name> -c launch_config.json`

## Self Build Images

At the root of the github repo:

`docker build . -t <ecr-image-name>:<tag>` 

`docker push <ecr-image-name>:<tag>`

`wandb launch -d <ecr-image-name>:<tag> -e <entity> -p <project> -q <queue-name>`   

The agent will receive the enqueued job, and start the image on SageMaker.

Workspace example: [https://wandb.ai/kylegoyette/sagemaker-ddp?nw=nwuserkylegoyette](https://wandb.ai/kylegoyette/sagemaker-ddp?nw=nwuserkylegoyette)

# Iterating:

## Using code artifacts

1. Modify code locally
2. `wandb launch -u /path/to/root/of/code --entry-point "python train_dis.py" -e <entity> -p <project> -q <queue-name> -c launch_config.json`

### Using Git Repos

1. Modify code locally
2. Push to git repo
3. `wandb launch -u <git-uri> -g <branch-name> --entry-point "python train_dis.py" -e <entity> -p <project> -q <queue-name> -c launch_config.json`

### Changing The Entrypoint

Say you’d like to run a different file in your code base, for example `train.py` , this can be done in a few ways:

1. Modify the entrypoint option when you launch your code. This will make a new job
    1. `wandb launch -u <git-uri> -g <branch-name> --entry-point "python train.py" -e <entity> -p <project> -q <queue-name> -c launch_config.json`
2. Launch an existing job, but override the entrypoint:
    
    a. UI - From the Jobs Page Click “Launch” on your job. Then edit the `entry_point` field in the overrides JSON.
    
    ![Untitled](SageMaker%20Distributed%20Data%20Example%20df9ffe5c979c4931892d62a1b29e4411/Untitled.png)
    
    b. CLI
    
    Modify the entrypoint field in `launch_config.json`
    
    ```yaml
    {
        "overrides": {
            "run_config": {
                "epochs": 10,
                "batch_size": 32,
                "lr": 0.001
            },
            "entry_point": ["python", "new_entrypoint.py"],
            "args": []
        }
    }
    ```
    
    Then you can launch the job with the new entrypoint:
    
    `wandb launch -j <entity>/<project>/<job-name>:<version> -q <queue> -e <entity> -p <project> -c launch_config.json`
