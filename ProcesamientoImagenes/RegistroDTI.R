# Script para realizar el registro lineal y no lineal de las imágenes DTI

library(RNiftyReg)

source = "ADNI/"
destiny = "NuevosDatos_Nonlinear/"
atlas <-readNifti("../AtlasesSize/MNI152_T1_2mm_brain.nii.gz")
mask <- readNifti("../AtlasesSize/MNI152_T1_2mm_brain_mask.nii.gz")
image_paths = list.files(source, recursive = TRUE)
count = 0
for (image_path in image_paths){
  
  file_name = tail(strsplit(image_path, "/")[[1]], n=1)
  print(file_name)
  image <- readNifti(paste(source, image_path, sep= "") )
  result <- niftyreg(image, atlas, targetMask= mask , threads = 16)
  result <- niftyreg(image, atlas, scope= "nonlinear", init=forward(result), targetMask= mask, threads = 16)
  writeNifti(result$image,paste(destiny, file_name, sep= ""))
  
  print(count / length(image_paths)) 
  count = count + 1
}

print("Fin con AD")
source = "CN/"
destiny = "CN_Nonlinear/"
image_paths = list.files(source, recursive = TRUE)
count = 0
for (image_path in image_paths){
  file_name = tail(strsplit(image_path, "/")[[1]], n=1)
  print(file_name)
  image <- readNifti(paste(source, image_path, sep= "") )
  result <- niftyreg(image, atlas, targetMask= mask, threads = 16)
  result <- niftyreg(image, atlas, scope= "nonlinear", init=forward(result), targetMask= mask,  threads = 16)
  writeNifti(result$image,paste(destiny, file_name, sep= ""))
  print(count / length(image_paths))
  count = count + 1
}
print("Fin")
