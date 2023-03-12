#!/usr/bin/env Rscript
library(reticulate)
source('/home/felipe/Documentos/AttributedSBM-master/FitAttribute.R')

args = commandArgs(trailingOnly=TRUE)
# Create a file
file.create("predict.npy")
np <- import("numpy")
# data reading
att <- np$load(args[1])
net <- np$load(args[2])
Out <- FitAttribute(net,att,0)
np$save("predict.npy", Out$Comm)
