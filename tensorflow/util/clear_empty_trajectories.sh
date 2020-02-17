#!/bin/bash


# echo 'file_path = ' $1
IFS='_' read -r -a array <<< "$1"


if [ "${array[-2]}" == "${array[-3]}" ]; then
	rm -r $1
	echo "Cleared " $1
fi


