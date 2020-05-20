#!/bin/bash
host=$1

if [ -z "$host" ]
then
  host="3.88.127.179"
fi

rsync -e "ssh -i ~/projects/dfried_alfred.pem" -avzhi --keep-dirlinks --include="*/" --include="*.txt" --include="*.py" --include="*.sh" --include="*.html" --exclude="*" * ubuntu@${host}:alfred
