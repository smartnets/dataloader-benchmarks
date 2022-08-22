# ml-benchmarks

This is a project for benchmarking data-loaders, with an emphasis on over-the-network data loading.

# Configuration on AWS

## Downloading Docker image

Create and run the following script

```sh
# get-ecr.sh
echo "y" | docker image prune
$(aws ecr get-login --no-include-email --region us-east-1)
docker pull 776309576316.dkr.ecr.us-east-1.amazonaws.com/loader-benchmark:1
```

> **_TODO:_** fix the tags associated with the images

## Environmental Variables

The file `src/config.py` has many configuration options that might need to be changed based on the machine where one is working.
When working on AWS, those variables can be supplied by creating a `.env` file with the appropriate variables (remember the prefix).

An example of one such file:

```txt
MY_AWS_ACCESS_KEY_ID=<key to the datasets and results bucket>
MY_AWS_SECRET_ACCESS_KEY=<key to the dataset and results bucket>
MY_BUCKET_NAME=<name-of-your-bucket>
```

## Running the docker container

The following script can be used to run any command inside the docker container

```sh 
# run.sh
docker run \
     --rm \
     -it \
     --gpus device=0 \
     --cap-add sys_ptrace \
     --privileged \
     --env-file .env \
     -v benchmarks-datasets:/home/worker/workspace/datasets \
     -v benchmarks-results:/home/worker/workspace/results \
     776309576316.dkr.ecr.us-east-1.amazonaws.com/loader-benchmark:1 \
     "$@"
```

> **_NOTE:_** beware of the hardcoded gpus in the command (device=0), as well as the image tag.

For example `./run.sh bash` will log in to the container.


# Running locally

## Setting up the S3 bucket

1. Run the script `./scripts/s3_minio.sh` to start the server.
2. Log into the dashboard (port 9001) and create a bucket.
3. Create a file devcontainer.env that might look as follows:

```sh
# devcontainer.env
MY_S3_ENDPOINT=http://<minio_ip>:9000
MY_BUCKET_NAME=<name-of-your-bucket>
```


# Running experiments

For each experiment, we first need to prepare its dataset. For this we run the following command:

```sh
python src/datasets/prepare.py --library <library-name> [--remote]
```
where the `--remote` flag is optional (only if loading the dataset remotely).


## CIFAR 10

Running all CIFAR10 experiments:

### Preparing the datasets

```sh
python src/datasets/prepare.py --library pytorch
python src/datasets/prepare.py --library ffcv
python src/datasets/prepare.py --library hub
python src/datasets/prepare.py --library hub --remote
python src/datasets/prepare.py --library webdataset
python src/datasets/prepare.py --library webdataset --remote
python src/datasets/prepare.py --library torchdata

```

> **_NOTE_:** webdataset --remote requires s3cmd that needs to be manually configured. For this, ssh into the container and run s3cmd --configure. Some important points. 1) If testing locally, answer "no" to HTTPS. 2) The test might fail, in any case answer yes to save the configuration. 3) In my experience using %(bucket) as the url format works fine.

### Running the benchmarks

```sh
export E=100
export PS=20
python src/run.py --epochs $E --profiling-epochs 0 10 30 50 --profiling-steps $PS --library pytorch
python src/run.py --epochs $E --profiling-epochs 0 10 30 50 --profiling-steps $PS --library ffcv
python src/run.py --epochs $E --profiling-epochs 0 10 30 50 --profiling-steps $PS --library hub
python src/run.py --epochs $E --profiling-epochs 0 10 30 50 --profiling-steps $PS --library hub --remote
python src/run.py --epochs $E --profiling-epochs 0 10 30 50 --profiling-steps $PS --library webdataset
python src/run.py --epochs $E --profiling-epochs 0 10 30 50 --profiling-steps $PS --library webdataset --remote
python src/run.py --epochs $E --profiling-epochs 0 10 30 50 --profiling-steps $PS --library torchdata

```

## COCO

Running all COCO experiments:

### Preparing the datasets

```sh
python src/datasets/prepare.py --dataset coco --library pytorch
python src/datasets/prepare.py --dataset coco --library webdataset
python src/datasets/prepare.py --dataset coco --library hub

```

> **_NOTE_:** webdataset --remote requires s3cmd that needs to be manually configured. For this, ssh into the container and run s3cmd --configure. Some important points. 1) If testing locally, answer "no" to HTTPS. 2) The test might fail, in any case answer yes to save the configuration. 3) In my experience using %(bucket) as the url format works fine.

### Running the benchmarks

```sh
export E=2
export PS=20
export BS=2
export LW=2
python src/run.py --epochs $E --profiling-epochs 0 5 --profiling-steps $PS --batch-size $BS --loader-workers $LW --dataset coco --library pytorch
python src/run.py --epochs $E --profiling-epochs 0 5 --profiling-steps $PS --batch-size $BS --loader-workers $LW --dataset coco --library hub
python src/run.py --epochs $E --profiling-epochs 0 5 --profiling-steps $PS --batch-size $BS --loader-workers $LW --dataset coco --library webdataset

```

## Filtering Functionality

### Prepare datasets

This is the same procedure as used for cifar and coco.

### Running the benchmarks

```sh
export E=10
python src/run.py --epochs $E --filtering --library pytorch
python src/run.py --epochs $E --filtering --library hub
python src/run.py --epochs $E --filtering --library torchdata

```

## Comparing performance versus workers and batch size

```sh
#!/usr/bin/env bash
# run_exp.sh

E=2
libs=( "pytorch" "hub" "webdataset" "ffcv" "torchdata")

for i in {1..3};
do
  for lib in ${libs[@]}
    do
      python src/run.py --library $lib --loader-workers $i
    done
done
```


# Configuration of GOOGLE API

It is important to add the GOOGLE_API 


# Implemented Libraries and Datasets

|          |           | Pytorch | FFCV | Hub | Hub3 | Torchdata | Webdataset | Squirrel |
| -------- | --------- | ------- | ---- | --- | ---- | --------- | ---------- | -------- |
| CIFAR-10 | default   | ✅      | ✅   | ✅  | ✅   | ✅        | ✅         |  ✅        |
|          | remote    | ❌      | ✅   | ✅  | ❌   | ❌        | ✅         |  ❓        |
|          | filtering | ✅      | ❓   | ✅  | ❌   | ✅        | ✅         |  ❓        |
|          | multi-gpu | ❓      | ❓   | ❓  | ❌   | ❓        | ❓         |  ❓        |
| RANDOM   | default   | ✅      | ✅   | ✅  | ✅   | ✅        | ✅         |  ✅        |
|          | remote    | ❌      | ✅   | ✅  | ❌   | ❌        | ✅         |  ❓        |
|          | filtering | ✅      | ❓   | ✅  | ❓   | ✅        | ✅         |  ❓        |
|          | multi-gpu | ❓      | ❓   | ❓  | ❌   | ❓        | ❓         |  ❓        |
| CoCo     | default   | ✅      | ❌   | ✅  | ❓   | ✅        | ✅         |  ✅        |
|          | remote    | ❌      | ❌   | ✅  | ❓   | ❌        | ✅         |  ❓        |
|          | filtering | ✅      | ❌   | ✅  | ❓   | ✅        | ✅         |  ❓        |
|          | multi-gpu | ❓      | ❓   | ❓  | ❌   | ❓        | ❓         |  ❓        |
