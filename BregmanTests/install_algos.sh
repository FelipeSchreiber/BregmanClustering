#!/bin/bash
rm -rf ./AttributedSBM
rm -rf ./CSBM
rm -rf ./data
git clone https://github.com/glmbraun/CSBM/
git clone https://github.com/stanleyn/AttributedSBM.git
git clone https://github.com/bkamins/ABCDGraphGenerator.jl.git
git clone https://github.com/MartijnGosgens/validation_indices
cd ./ABCDGraphGenerator.jl/utils/
julia install.jl