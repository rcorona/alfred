#!/bin/bash
host=$1

if [ -z "$host" ]
then
  host="fromage"
fi

rsync -avzhi --keep-dirlinks --include="*/" --include="*.txt" --include="*.py" --include="*.sh" --include="*.html" --exclude="*" * ${host}:/data/dfried/projects/alfred/alfred
