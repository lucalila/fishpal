#!/bin/sh

for f in exps/mrpc/*.sh; do
  if [ "$f" == "$0" ]; then
    continue
  else
    echo "running: $f"
    bash "$f" -H
  fi
done