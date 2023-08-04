#!/usr/bin/env Rscript
library(reticulate)
source('./AttributedSBM/FitAttribute.R')
args = commandArgs(trailingOnly=TRUE)
# Create a file
file.create("predict.npy")
np <- import("numpy")
print(args)
# data reading
att <- np$load(args[1])
net <- np$load(args[2])
z_init <- np$load(args[3])
Out <- FitAttribute(net,att,0,z_init)
np$save("predict.npy", Out$Comm)
