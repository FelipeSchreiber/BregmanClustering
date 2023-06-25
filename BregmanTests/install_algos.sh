#!/bin/bash
rm -rf ./AttributedSBM
rm -rf ./CSBM
rm -rf ./data
git clone https://github.com/glmbraun/CSBM/
git clone https://github.com/stanleyn/AttributedSBM.git
git clone https://github.com/bkamins/ABCDGraphGenerator.jl.git
pwd=$(pwd)
cd /
wget https://julialang-s3.julialang.org/bin/linux/x64/1.8/julia-1.8.1-linux-x86_64.tar.gz
tar zxvf julia-1.8.1-linux-x86_64.tar.gz -C /usr/local --strip-components 1
cd $pwd 
cd ./ABCDGraphGenerator.jl/utils/
julia install.jl