data_roi <- read.csv("causalidad_roi_nuevas.csv")
p_value <- 1 -pnorm(abs(data_roi[["X_Y"]]-data_roi[["Y_X"]]), sd = sqrt(abs(data_roi[["sigma2"]])))  
final_df_roi <- cbind(data_roi, p_value)
final_df_roi[order(final_df_roi["p_value"]),][c("ROI", "p_value")]

data_genes <- read.csv("causalidad_genes_probs.csv")
p_value <- 1 - pnorm(abs(data_genes[["X_Y"]]-data_genes[["Y_X"]]), sd = sqrt(abs(data_genes[["sigma2"]])))
final_df_genes <- cbind(data_genes, p_value)
final_df_genes_sorted <- final_df_genes[order(final_df_genes["p_value"]),]
final_df_genes_sorted
final1 <- head(final_df_genes_sorted)
final1
