aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 776309576316.dkr.ecr.us-east-1.amazonaws.com
docker tag yins/loader-benchmark:latest 776309576316.dkr.ecr.us-east-1.amazonaws.com/loader-benchmark:latest


