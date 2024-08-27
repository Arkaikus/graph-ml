# Install and load necessary packages
if (!requireNamespace("ggplot2", quietly = TRUE)) {
  install.packages("ggplot2")
}
if (!requireNamespace("poweRlaw", quietly = TRUE)) {
  install.packages("poweRlaw")
}
library(ggplot2)
library(poweRlaw)

# Set seed for reproducibility
set.seed(123)

# Generate dummy data for each distribution
# Gaussian (Normal) distribution
gaussian_data <- rnorm(1000, mean = 10, sd = 2)

# Poisson distribution
poisson_data <- rpois(1000, lambda = 5)

# Power-law distribution (Zipf distribution)
# Generate power-law distributed data using poweRlaw package
xmin <- 1
alpha <- 2
powerlaw_data <- rpldis(1000, xmin = xmin, alpha = alpha)

# Create data frames for plotting
gaussian_df <- data.frame(value = gaussian_data, distribution = "Gaussian")
poisson_df <- data.frame(value = poisson_data, distribution = "Poisson")
powerlaw_df <- data.frame(value = powerlaw_data$x, distribution = "Power-law")

# Combine data frames
combined_df <- rbind(gaussian_df, poisson_df, powerlaw_df)

# Plotting using ggplot2
plot <- ggplot(combined_df, aes(x = value, fill = distribution)) +
  geom_density(alpha = 0.6) +  # Density plot for each distribution
  facet_wrap( ~ distribution, scales = "free") +  # Separate panels for each distribution
  labs(title = "Density plots of Gaussian, Poisson, and Power-law distributions", x = "Value", y = "Density") +
  theme_minimal()

# Display the plot
print(plot)
