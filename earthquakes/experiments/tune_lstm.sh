quakes lstm tune --samples 10 --classify t --metric accuracy --mode max
quakes lstm tune --samples 10 --classify t --networkx t --node-size 50 --metric accuracy --mode max
quakes lstm tune --samples 10 --classify t --networkx t --node-size 100 --metric accuracy --mode max
quakes lstm tune --samples 10 --classify t --networkx t --node-size 150 --metric accuracy --mode max

quakes lstm test --classify t --metric accuracy --mode max

