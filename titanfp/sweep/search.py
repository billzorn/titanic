"""Common code for parameter search."""


# Goal: find a distribution of "good" configurations according to a set of metrics

# Kinds of metrics:
# - quality
# - cost
# Ways to search:
# - quality must be with in bound, find best cost (the most common way)
# - cost must be wtihin bound, do as well as you can

# This means each metric either has a bound or doesn't
# When evaluating a configuration, there are two modes:
# - bound metric is satisfied
#   - improve "cost" metrics
# - bound metric not satisfied
#   - improve bound metric

# Search procedure
#
# Shotgun: pick a random configuration, based on a set of annotation sites and format hints
# Evaluate: check the configuration's evaluation
# Hillclimb: Look at nearby configurations (move each annotation [up to]? k steps)
#   Among those configurations, look for the "most passing" configuration
#   - if there is a passing configuration, use the cheapest
#   - otherwise, use the least bad
#   probably move each point at least k1 steps, and at most k2 steps, in each direction
#   - start by moving k1
#   - then, if there isn't a passing configuration, keep going up to k2?
#   - no, just search fixed k in each direction, it's simpler
#   Now update our evaluated starting config and keep climbing

# Keeping a set of "good" configurations
#
# We have a set of metrics, some of which have hard bounds
# Most of them we can also order
# Our goal is to only consider options that satisfy all the bounds; of those we want the best
# Obviously, only keep configurations that satisfy the bounds
# Of those that do, look at all other cost metrics we care about
# keep a "frontier set" that are each better at some subset of metrics than any other member of frontier
# search until no point is near a point that can be improved, and after a certain number of random inits


# functions we need:
# init - randomly pick a starting point, based on a precision hint
# next/prev on types - get higher and lower precision, in a sequence
# compare metric - given two values, which is better
# bound metric - given a value, is it ok

# there's an optimization frontier - things might split

# Hillclimbing with random restarts

# cost metric isn't about the chip - we have proxies
# spin it as an advantage!
