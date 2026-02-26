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
earthquake_data <- earthquake_data[earthquake_data$magnitude > 0, ]
# descriptores max min media mediana varianza desviacion
# density of magnitudes, descriptores de la variable, media, varianza, tipo de distribucion, sesgada a la izq
# ggplot(earthquake_data, aes(x = mag)) +
#   geom_density(fill = "steelblue", alpha = 0.5) +
#   labs(title = "Magnitude Distribution", x = "Magnitude", y = "Density") +
#   scale_x_continuous(breaks = seq(floor(min(earthquake_data$mag)), ceiling(max(earthquake_data$mag)), by = 0.1))
# Define the power law function
power_law <- function(x, alpha = 2.5, C = 1) {
  ifelse(x > 0, C * x ^ (-alpha), 0)
}

# Function to plot distributions with annotations
# Function to plot distributions with annotations
plot_distribution <- function(data,
                              feature,
                              plot_color,
                              brks = 0.4,
                              metrics_position = "top right",
                              metrics_xpad = 0.02,
                              metrics_ypad = 0.05) {
  # Calculate basic statistics
  feature_data <- data[[feature]]
  median_val <- median(feature_data)
  mean_val <- mean(feature_data)
  sd_val <- sd(feature_data)
  var_val <- var(feature_data)
  min_val <- min(feature_data)
  max_val <- max(feature_data)
  
  metrics <- paste(
    paste("Min:", round(min_val, 2)),
    "\n",
    paste("Max:", round(max_val, 2)),
    "\n",
    paste("Mean μ:", round(mean_val, 2)),
    "\n",
    paste("SD σ:", round(sd_val, 2)),
    "\n",
    paste("Variance:", round(var_val, 2)),
    "\n"
  )
  
  # Calculate the Gutenberg-Richter b-value (simplified)
  if (feature == "magnitude") {
    b_value <- 1 / (mean_val - (min_val - 0.05))
    metrics <- paste(metrics, paste("b-value:", round(b_value, 2)), "\n")
  }
  
  # Set position for metrics box
  position <- tolower(metrics_position)
  if (position == "top") {
    x_pos <- 0.5
    y_pos <- 0.95
    hjust_val <- 0.5
    vjust_val <- 1
  } else if (position == "top left") {
    x_pos <- 0.05
    y_pos <- 0.95
    hjust_val <- 0
    vjust_val <- 1
  } else if (position == "top right") {
    x_pos <- 0.95
    y_pos <- 0.95
    hjust_val <- 1
    vjust_val <- 1
  } else if (position == "bottom") {
    x_pos <- 0.5
    y_pos <- 0.05
    hjust_val <- 0.5
    vjust_val <- 0
  } else if (position == "bottom left") {
    x_pos <- 0.05
    y_pos <- 0.05
    hjust_val <- 0
    vjust_val <- 0
  } else if (position == "bottom right") {
    x_pos <- 0.95
    y_pos <- 0.05
    hjust_val <- 1
    vjust_val <- 0
  } else {
    stop(
      "Invalid position. Choose 'top', 'top left', 'top right', 'bottom', 'bottom left', or 'bottom right'."
    )
  }
  
  # Create the plot
  plot <- ggplot(data, aes_string(x = feature)) +
    geom_density(
      aes(y = ..density.., fill = "Data Density"),
      alpha = 0.5,
      color = "black"
    ) +
    stat_function(
      fun = dnorm,
      args = list(mean = mean_val, sd = sd_val),
      aes(color = "Gaussian"),
      size = 1
    ) +
    labs(
      title = paste("Distribution of", tools::toTitleCase(feature)),
      x = tools::toTitleCase(feature),
      y = "Density",
      size = 10
    ) +
    scale_x_continuous(breaks = seq(floor(min_val), ceiling(max_val), by = brks)) +
    geom_vline(
      aes(xintercept = median_val),
      color = "red",
      linetype = "dotted",
      size = 1,
      alpha = 0.5
    ) +
    geom_vline(
      aes(xintercept = median_val + sd_val),
      color = "black",
      linetype = "dotted",
      size = 1,
      alpha = 0.5
    ) +
    geom_vline(
      aes(xintercept = median_val - sd_val),
      color = "black",
      linetype = "dotted",
      size = 1,
      alpha = 0.5
    ) +
    annotate(
      "text",
      x = median_val,
      y = 0,
      label = paste("μ:", round(median_val, 2)),
      color = "red",
      vjust = -2.5
    ) +
    annotate(
      "text",
      x = median_val + sd_val,
      y = 0,
      label = paste("μ + σ:", round(median_val + sd_val, 2)),
      color = "black",
      vjust = -1
    ) +
    annotate(
      "text",
      x = median_val - sd_val,
      y = 0,
      label = paste("μ - σ:", round(median_val - sd_val, 2)),
      color = "black",
      vjust = -1
    ) +
    theme(legend.position = "top",
          legend.text = element_text(size = 10)) +
    scale_fill_manual(values = c("Data Density" = plot_color)) +
    scale_color_manual(values = c("Gaussian" = "blue")) +
    labs(color = "Distributions", fill = "Density") +
    annotation_custom(grob = grobTree(
      rectGrob(
        x = unit(x_pos, "npc"),
        y = unit(y_pos, "npc"),
        width = unit(1.05, "strwidth", metrics),
        # Increased from 5 to 5.5
        height = unit(1.2, "lines") * length(strsplit(metrics, "\n")[[1]]),
        # Increased from 1.5 to 1.7
        hjust = hjust_val,
        vjust = vjust_val,
        gp = gpar(
          fill = "white",
          alpha = 0.8,
          col = "black"
        )
      ),
      textGrob(
        label = metrics,
        x = unit(x_pos - metrics_xpad, "npc"),
        # Adjust x position
        y = unit(y_pos - metrics_ypad, "npc"),
        # Adjust y position
        hjust = hjust_val,
        vjust = vjust_val,
        gp = gpar(fontsize = 10)
      )
    ))
  
  return(plot)
}

magplot <- plot_distribution(earthquake_data, "magnitude", "orange")
depth_plot <- plot_distribution(earthquake_data, "depth", "yellow", brks = 20)
lat_plot <- plot_distribution(
  earthquake_data,
  "latitude",
  "steelblue",
  brks = 0.7,
  metrics_position = "top left",
  metrics_xpad = -0.02
)
long_plot <- plot_distribution(
  earthquake_data,
  "longitude",
  "green",
  brks = 0.6,
  metrics_position = "top left",
  metrics_xpad = -0.02
)

print((magplot | depth_plot) / (lat_plot | long_plot))