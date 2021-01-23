# Control de calidad de los datos genéticos

library(plinkQC)

indir <- file.path("GeneticData/GeneticData/")
qcdir <- file.path("GeneticData/Results/")
path2plink <- file.path("plink/plink")
name <- "mergedADNI"


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

