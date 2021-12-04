#! /bin/bash

cd /cygdrive/c/projects/nrel-presentation/data/raw

while true
    do
        wget --user=itsftp --password='IzzyR^les#1' ftp://ftp.its.nv.gov/external/FASTrealtime.xml 2>&1 1>/dev/null
        mv FASTrealtime.xml $(date +%Y%m%d_%H%M%S)_FASTrealtime.xml
        sleep 56
    done

