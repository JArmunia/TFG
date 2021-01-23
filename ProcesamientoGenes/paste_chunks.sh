#!/bin/bash

# Script para unir cada chunck de la imputación

set -e
set -u
refdir="1000GImpute2";
for chr in {1..22}; do
    echo Empieza
    namefile="ImputationChunks/chr${chr}";
    maxPos=$(gawk '$1!="position" {print $1}' ${refdir}/genetic_map_chr${chr}_combined_b37.txt | sort -n | tail -n 1);
    nrChunk=$(expr ${maxPos} "/" 5000000);
    nrChunk2=$(expr ${nrChunk} "+" 1);
       
    for chunk in $(seq 1 $nrChunk2); do
        file="${namefile}.chunk${chunk}.impute2"
        if test -f "$file"; then            
            cat ${file} >> ImputeDosages/chr${chr}.impute2.dosage;  
        else 
            echo "No existe ${file}" 
        fi   
    done
    echo "Acaba ${chr}"
done