# WAIT! change the experiment folders
quakes lstm tune --quantiles 2 --samples 10 --metric accuracy --mode max -ex ~/ray_results/ClassificationTrainable_2024-11-28_13-26-04
quakes lstm tune --quantiles 3 --samples 10 --metric accuracy --mode max -ex ~/ray_results/ClassificationTrainable_2024-11-28_15-03-20
quakes lstm tune --quantiles 4 --samples 10 --metric accuracy --mode max -ex ~/ray_results/ClassificationTrainable_2024-11-28_15-05-48