# Script para realizar el phasing del genoma

for chr in {1..22}; do
    reffile="reference/glist-hg19-chr${chr}";
    cat glist-hg19 | grep -E "^${chr} " | sort -k2 -n | cut -d " " -f "2 3 4" > ${reffile} 
done

for chr in {1..22}; do
    reffile="1000GImpute2/genetic_map_chr${chr}_combined_b37.txt";
    phasedfile="phased-chr/chr${chr}.phased";
    ./shapeit.v2.904.3.10.0-693.11.6.el7.x86_64/bin/shapeit \
        -B chr-bims/Subjects_Filtered_DS-updated-chr${chr} \
        -M ${reffile} \
        -O ${phasedfile} \
        --thread 16
done

