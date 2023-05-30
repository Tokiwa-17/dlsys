#!/bin/sh

set -x
#if [ -d build ]; then
#  rm -r build
#fi

cd build

cmake .. && make

cd ..

./build/mytest