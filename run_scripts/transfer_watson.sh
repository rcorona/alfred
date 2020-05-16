#!/bin/bash
host=$1

if [ -z "$host" ]
then
  host="watson"
fi

rsync -avzhi --keep-dirlinks --include="*/" --include="*.txt" --include="*.py" --include="*.sh" --include="*.html" --exclude="*" * ${host}:/work/dfried/projects/alfred
