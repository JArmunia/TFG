# Script para realizar la segmentación y reducción de la dimensionalidad mediante FPCA de las imágenes DTI

library(RNiftyReg)
library(FPCA3D)
library(imager)
library(plot3D)

# Function to load the brain images
load_images <- function(source, path){
  images <- readNifti(paste(source, path, sep= ""))
  #images[is.na(images)] <- 0
  f <- function(im){
    im[is.na(im)] <- 0
    return(im)
  }
  
  return(lapply(images, f))
}

# Function to apply the mask to the image
apply_mask <- function(img, mask){
  img[!mask] <- 0
  return(img)
}

# Function to crop the images
crop_images <- function(images){
  # Find cropping coordinates
  x_min <- 0
  x_max <- dim(images)[1]
  y_min <- 0
  y_max <- dim(images)[2]
  z_min <- 0
  z_max <- dim(images)[3]
  
  compare_img <- images[,,,1] > 0
  
  for(i in 1:dim(compare_img)[1]){
    if(any(compare_img[i, , ])){
      x_min <- i
      break
    }
  }
  for(i in dim(compare_img)[1]:1){
    if(any(compare_img[i, , ])){
      x_max <- i
      break
    }
  }
  for(i in 1:dim(compare_img)[2]){
    if(any(compare_img[, i, ])){
      y_min <- i
      break
    }
  }
  for(i in dim(compare_img)[2]:1){
    if(any(compare_img[, i, ])){
      y_max <- i
      break
    }
  }
  for(i in 1:dim(compare_img)[3]){
    if(any(compare_img[, , i])){
      z_min <- i
      break
    }
  }
  for(i in dim(compare_img)[3]:1){
    if(any(compare_img[, , i])){
      z_max <- i
      break
    }
  }
  
  x_len <- length(x_min:x_max)
  y_len <- length(y_min:y_max)
  z_len <- length(z_min:z_max)
  
  output <- images[x_min:x_max, y_min:y_max, z_min:z_max,]
  
  return(output)
  
}


fpca_complete <- function(images, mask, output_file){
  
  masked_imgs <- lapply(images, apply_mask, mask)
  
  
  for(i in seq(1, length(masked_imgs))){
    masked_imgs[[1]] <- array(masked_imgs[[1]], dim = c(91,109,91))
  }
  images_array <- array(rep(0, 91*109*91*370), dim= c(91,109,91,370))
  dim(images_array)
  for(i in seq(1, 370)){
    images_array[,,,i] <- masked_imgs[[i]]
  }
  
  
  
  masked_imgs <- 0
  images <- 0
  mask <- 0
  images_array <- crop_images(images_array)
  
  # Normalizamos
  mu <- mean(images_array)
  sigma <-sd(images_array)
  
  images_array <- (images_array - mu)/ sigma
  
  mean(images_array)
  sd(images_array)
  
  # Realizamos FPCA
  start.time <- Sys.time()
  fpcas <- FPCA_3D_score(images_array, 0.95)
  end.time <- Sys.time()
  time.taken <- end.time - start.time
  print("Tiempo total: ")
  print(time.taken)
  
  # Guardamos las imágenes
  df_fpcas <- as.data.frame(fpcas)
  df_fpcas$file <- image_paths
  write.csv(df_fpcas, output_file, row.names = TRUE)
}


source = "E:Corrected_FA/ALL_DATA/"
# Ventricle mask 
# mask_path = "E:Corrected_FA/Atlasesfsl/MNI152_T1_2mm_VentricleMask.nii.gz"

# Temporal mask

masks_path <- "AtlasesFSL/MNI/MNI-prob-2mm.nii.gz"

ROIs = c("Caudate", 
         "Cerebellum", 
         "Frontal_Lobe", 
         "Insula", 
         "Occipital_Lobe", 
         "Parietal_Lobe", 
         "Putamen", 
         "Temporal_Lobe", 
         "Thalamus")

masks <- array(readNifti(masks_path), dim=c(91,109,91,9)) > 0
#mask_temporal <- mask_temporal[,,,8] # Temporal lobe
image_paths = list.files(source)

images <- load_images(source, image_paths)
#
#right_temporal_mask <- array(rep(FALSE, 46 * 91 * 109), dim = c(91,109,91))
#
#right_temporal_mask[1:46,,] <- mask_temporal[1:46,,]
#
#left_temporal_mask <- array(rep(FALSE, 46 * 91 * 109), dim = c(91,109,91))
#
#left_temporal_mask[46:91,,] <- mask_temporal[46:91,,]
#

for(i in 1:9){
  start_time = Sys.time()
  fpca_complete(images, masks[,,,i], paste("FPCAs/FPCAs_",ROIs[i], ".csv"))
  print(paste(Sys.time()- start_time,  ROIs[i]))
}
#fpca_complete(images, right_temporal_mask, "FPCAS_Right_Temporal_Lobe.csv")
#fpca_complete(images, left_temporal_mask, "FPCAS_Left_Temporal_Lobe.csv")

