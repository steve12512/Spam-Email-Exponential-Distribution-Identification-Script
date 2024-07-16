Read the assignment pdf for more info.
This script reads a file containing the time intervals during which a message(spam email) was sent.
The rhythm of the emission of emails follows the exponential distribution, with varying levels of the λ variable, during different intervals.
Each different distribution is considered a different state of message emitting.
The purpose of this script is to identify the start and the elapse of different exponential distribution states and mark the intervals during which these take place.
The user can either run the viterbi algorithm or the bellman ford version which uses a graph and traverses it using the homonymous method.
You can either adjust the s, g variables or else their default values will be used, whereas the -d option provides more printing details
