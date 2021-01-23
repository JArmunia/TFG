#!/bin/bash

# Script para realizar la imputación de cada cromosoma

start_time=`date +%s`
chr=$1
refdir="1000GImpute2";

namefile="phased-chr/chr${chr}.phased.haps";
maxPos=$(gawk '$1!="position" {print $1}' ${refdir}/genetic_map_chr${chr}_combined_b37.txt | sort -n | tail -n 1);
echo ${maxPos}
nrChunk=$(expr ${maxPos} "/" 5000000);     
nrChunk2=$(expr ${nrChunk} "+" 1);     
start="0";     
for chunk in $(seq 1 $nrChunk2); do            
    endchr=$(expr $start "+" 5000000);            
    startchr=$(expr $start "+" 1); 
    outputFile="ImputationChunks/chr${chr}.chunk${chunk}.impute2"    
    if test -f "$outputFile"; then
        echo Existe
    else      
        ./impute2dynamic/impute_v2.3.2_x86_64_dynamic/impute2 \
            -known_haps_g ${namefile} \
            -m ${refdir}/genetic_map_chr${chr}_combined_b37.txt \
            -h ${refdir}/1000GP_Phase3_chr${chr}.hap.gz \
            -l ${refdir}/1000GP_Phase3_chr${chr}.legend.gz \
            -int ${startchr} ${endchr} -Ne 20000 -o \
            ImputationChunks/chr${chr}.chunk${chunk}.impute2;  
    fi
    start=${endchr};     
done

end=`date +%s`
runtime=$((end-start_time))
echo ${runtime} > runtime_chr${chr}.txt
echo ${start_time} >> runtime_chr${chr}.txt
echo ${end} >> runtime_chr${chr}.txt