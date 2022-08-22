#!/bin/bash
# Installs S3 sub module of AWS C++ SDK
set -e
apt update
apt-get install -y software-properties-common
apt update
add-apt-repository -y ppa:ubuntu-toolchain-r/test
apt update && apt upgrade && apt dist-upgrade
apt install -y g++-11 libstdc++6
apt install cmake -y
apt install zlib1g-dev libssl-dev libcurl4-openssl-dev -y
set +e
rm -rf aws-sdk-cpp
set -e
git clone --recurse-submodules https://github.com/aws/aws-sdk-cpp
cd aws-sdk-cpp
mkdir build
cd build
cmake .. -DBUILD_ONLY=s3 -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_SHARED_LIBS=ON
make
make install
