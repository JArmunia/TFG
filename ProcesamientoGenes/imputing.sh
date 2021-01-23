#!/bin/bash

# Script para imputar todo el genoma
for chr in {1..24}; do
    ./impute_chr.sh ${chr} 
done


