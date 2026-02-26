# Install required packages if not already installed
if (!require("ggplot2"))
  install.packages("ggplot2")
if (!require("dplyr"))
  install.packages("dplyr")
if (!require("patchwork"))
  install.packages("patchwork")
if (!require("grid"))
  install.packages("grid")
# Load required packages
library(ggplot2)
library(dplyr)
library(patchwork)
library(grid)
# library(lubridate)

# Assuming your CSV file is named "earthquake_data.csv" and is located in your working directory
original_earthquake_data <- read.csv("~/Documents/graph-ml/earthquakes/csv/9923ce9a42736848b544e335a4d7c5fb.csv")
original_earthquake_data <- original_earthquake_data[, c("latitude", "longitude", "depth", "mag", "time")]
# original_row_count <- nrow(original_earthquake_data)
print(paste("Number of records", nrow(original_earthquake_data)))

# Remove rows with any missing values
earthquake_data <- original_earthquake_data[complete.cases(original_earthquake_data), ]
print(paste("Number of usable records", nrow(earthquake_data)))

# Convert columns to numeric
earthquake_data$latitude <- as.numeric(as.character(earthquake_data$latitude))
earthquake_data$longitude <- as.numeric(as.character(earthquake_data$longitude))
earthquake_data$depth <- as.numeric(as.character(earthquake_data$depth))
earthquake_data$magnitude <- as.numeric(as.character(earthquake_data$mag))

# descriptores max min media mediana varianza desviacion
# density of magnitudes, descriptores de la variable, media, varianza, tipo de distribucion, sesgada a la izq

to_latex_table <- function(data) {
  # Initialize an empty vector to store LaTeX rows
  latex_rows <- c()
  
  # Define the features to analyze
  features <- c("latitude", "longitude", "depth", "magnitude")
  
  # Loop through each feature
  for (feature in features) {
    feature_data <- data[[feature]]
    
    # Calculate basic statistics
    median_val <- median(feature_data)
    mean_val <- mean(feature_data)
    sd_val <- sd(feature_data)
    var_val <- var(feature_data)
    min_val <- min(feature_data)
    max_val <- max(feature_data)
    
    # Create a LaTeX row for the current feature
    latex_row <- sprintf(
      "%s & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f \\",
      feature, min_val, max_val, median_val, mean_val, sd_val, var_val, max_val - min_val
    )
    
    # Add the row to the vector of LaTeX rows
    latex_rows <- c(latex_rows, latex_row)
  }
  
  # Combine all rows into a single LaTeX table
  latex_table <- c(
    "\\begin{tabular}{|l|r|r|r|r|r|r|r|}",
    "\\hline",
    "Feature & Min & Max & Median & Mean & SD & Variance & Range \\\\",
    "\\hline",
    latex_rows,
    "\\hline",
    "\\end{tabular}"
  )
  
  # Join all lines of the table into a single string
  return(paste(latex_table, collapse = "\n"))
}

table <- to_latex_table(earthquake_data)
print(table)