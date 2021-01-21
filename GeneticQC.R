library(plinkQC)

indir <- file.path("GeneticData/GeneticData/")
qcdir <- file.path("GeneticData/Results/")
path2plink <- file.path("plink/plink")
name <- "mergedADNI"

## Individuals with discordant sex information
#
#fail_sex <- check_sex(indir=indir, qcdir=qcdir, name=name, interactive=TRUE,
#                      verbose=TRUE, path2plink=path2plink)
#
## Individuals with outlying missing genotype and/or heterozygosity rates
#fail_het_imiss <- check_het_and_miss(indir=indir, qcdir=qcdir, name=name,
#                                     interactive=TRUE, path2plink=path2plink,
#                                     imissTh = 0.1)
#
## Related individuals - Identitiy By Descent
#exclude_relatedness <- check_relatedness(indir=indir, qcdir=qcdir, name=name,
#                                         interactive=TRUE,
#                                         path2plink=path2plink, 
#                                         highIBDTh = 0.185)
#
## Markers with excessive missingness rate
#fail_snpmissing <- check_snp_missingness(indir=indir, qcdir=qcdir, name=name,
#                                         interactive=TRUE,
#                                         path2plink=path2plink,
#                                         showPlinkOutput=FALSE, 
#                                         lmissTh = 0.1)
#
## Markers with deviation from HWE
#fail_hwe <- check_hwe(indir=indir, qcdir=qcdir, name=name, interactive=TRUE,
#                      path2plink=path2plink, showPlinkOutput=FALSE, 
#                      hweTh = 1e-06)

fail_individuals <- perIndividualQC(indir=indir, qcdir=qcdir, name=name,
                                    path2plink=path2plink, imissTh = 0.1,
                                    highIBDTh = 0.185, 
                                    dont.check_ancestry = TRUE,
                                    interactive=TRUE, verbose=TRUE)


fail_markers <- perMarkerQC(indir=indir, qcdir=qcdir, name=name,
                            path2plink=path2plink,
                            lmissTh = 0.1,
                            hweTh = 1e-06,
                            verbose=TRUE, interactive=TRUE,
                            showPlinkOutput=FALSE)


# Create QC-ed dataset
Ids <- cleanData(indir=indir, qcdir=qcdir, name=name, path2plink=path2plink,
                 verbose=TRUE, showPlinkOutput=FALSE,  
                 filterAncestry = FALSE)

