#!/usr/bin/env bash

default_cuda_device=0
root_dir=/local/fever-common

ln -s $root_dir/data data

echo "start evidence retrieval"

python -m fever.evidence.retrieve \
    --index $root_dir/data/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz \
    --database $root_dir/data/fever/fever.db \
    --in-file $1 \
    --out-file /tmp/ir.$(basename $1) \
    --max-page 5 \
    --max-sent 5

echo "start prediction"
python -m allennlp.run predict \
    https://jamesthorne.co.uk/fever/${model:-fever-da}.tar.gz \
    /tmp/ir.$(basename $1) \
    --output-file /tmp/labels.$(basename $1) \
    --predictor fever \
    --include-package fever.reader \
    --cuda-device ${CUDA_DEVICE:-$default_cuda_device} \
    --silent

echo "prepare submission"
python -m fever.submission.prepare \
    --predicted_labels /tmp/labels.$(basename $1) \
    --predicted_evidence /tmp/ir.$(basename $1) \
    --out_file $2
