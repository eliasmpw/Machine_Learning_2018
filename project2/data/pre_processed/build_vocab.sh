#!/bin/bash

# Note that this script uses GNU-style sed. On Mac OS, you are required to first
#    brew install gnu-sed --with-default-names
cat tweets_pos_full_test.txt tweets_neg_full_test.txt | sed "s/ /\n/g" | grep -v "^\s*$" | sort | uniq -c > vocab_test.txt
