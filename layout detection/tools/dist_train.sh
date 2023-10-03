#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/train.py $CONFIG  ${@:3}
