#!/bin/bash
rm -rf ./AttributedSBM
rm -rf ./CSBM
rm -rf ./data
git clone https://github.com/glmbraun/CSBM/
git clone https://github.com/stanleyn/AttributedSBM.git
git clone https://github.com/bkamins/ABCDGraphGenerator.jl.git
# #---------------------------------------------------#
JULIA_VERSION="1.8.5" # any version ≥ 0.7.0
# JULIA_PACKAGES="IJulia BenchmarkTools PyCall PyPlot"
# JULIA_PACKAGES_IF_GPU="CUDA" # or CuArrays for older Julia versions
# JULIA_NUM_THREADS=4
# #---------------------------------------------------#

if [ -z `which julia` ]; then
  # Install Julia
  echo ">>>>>>>>>>>>>>>>>>>>INSTALLING JULIAAAAAAAAAAAA"
  JULIA_VER=`cut -d '.' -f -2 <<< "$JULIA_VERSION"`
  echo "Installing Julia $JULIA_VERSION on the current Colab Runtime..."
  BASE_URL="https://julialang-s3.julialang.org/bin/linux/x64"
  URL="$BASE_URL/$JULIA_VER/julia-$JULIA_VERSION-linux-x86_64.tar.gz"
  wget -nv $URL -O /tmp/julia.tar.gz # -nv means "not verbose"
  tar -x -f /tmp/julia.tar.gz -C /usr/local --strip-components 1
  rm /tmp/julia.tar.gz

#   # Install Packages
#   nvidia-smi -L &> /dev/null && export GPU=1 || export GPU=0
#   if [ $GPU -eq 1 ]; then
#     JULIA_PACKAGES="$JULIA_PACKAGES $JULIA_PACKAGES_IF_GPU"
#   fi
#   for PKG in `echo $JULIA_PACKAGES`; do
#     echo "Installing Julia package $PKG..."
#     julia -e 'using Pkg; pkg"add '$PKG'; precompile;"' &> /dev/null
#   done

#   # Install kernel and rename it to "julia"
#   echo "Installing IJulia kernel..."
#   LD_PRELOAD_ julia -e 'using IJulia; IJulia.installkernel("julia", env=Dict(
#       "JULIA_NUM_THREADS"=>"'"$JULIA_NUM_THREADS"'"))'
#   KERNEL_DIR=`julia -e "using IJulia; print(IJulia.kerneldir())"`
#   KERNEL_NAME=`ls -d "$KERNEL_DIR"/julia*`
#   mv -f $KERNEL_NAME "$KERNEL_DIR"/julia

#   echo ''
#   echo "Successfully installed `julia -v`!"
#   echo "Please reload this page (press Ctrl+R, ⌘+R, or the F5 key) then"
#   echo "jump to the 'Checking the Installation' section."
# fi
cd ./ABCDGraphGenerator.jl/utils/
julia install.jl