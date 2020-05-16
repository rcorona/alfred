#!/bin/bash
host=$1

if [ -z "$host" ]
then
  host="nlp1"
fi

rsync -avzhi --keep-dirlinks --include="*/" --include="*.txt" --include="*.py" --include="*.sh" --include="*.html" --exclude="*" * ${host}:/home/dfried/projects/alfred_dev
