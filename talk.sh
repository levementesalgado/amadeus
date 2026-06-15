#!/bin/bash
cd /root/projects/amadeus

output=$(./target/release/amadeus_m_train \
  --order=10 --temp=0.7 --maxlen=80 --explore=0.01 \
  --sat=1.0 --affect=a \
  --say="$1" 2>/dev/null)

echo "$output" | sed -n '/Amadeus:/s/.*Amadeus: //p'
