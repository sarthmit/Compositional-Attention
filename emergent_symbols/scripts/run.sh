#!/bin/bash

extras="$1"

./scripts/same_diff.sh tsf 0 8 1 0 "$extras"
./scripts/identity_rules.sh tsf 0 8 1 0 "$extras"
./scripts/RMTS.sh tsf 0 8 1 0 "$extras"
./scripts/dist3.sh tsf 0 8 1 0 "$extras"

values=( 1 2 4 6 8 )
for v in "${values[@]}"; do
  ./scripts/same_diff.sh comp_tsf 0 8 1 $v "$extras"
  ./scripts/identity_rules.sh comp_tsf 0 8 1 $v "$extras"
  ./scripts/RMTS.sh comp_tsf 0 8 1 $v "$extras"
  ./scripts/dist3.sh comp_tsf 0 8 1 $v "$extras"
done