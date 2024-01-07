#!/bin/sh

for f in exps/*.sh; do
  if [ "$f" == "$0" ]; then
    continue
  else
    echo "running: $f"
    bash "$f" -H
  fi
done