#!/bin/bash

LOG_FILE_PATH="$PWD/.logs/celery_sync_$(date +\%Y-\%m-\%d).log"
export PYTHONPATH=$PWD
poetry run celery -A core.queueing worker --without-gossip -P prefork --loglevel=INFO 2>&1 | tee -a $LOG_FILE_PATH
