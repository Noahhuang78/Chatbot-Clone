## Start Redis server ##
docker run -d --name MyRedisContainer -p 6379:6379 redis

## Start Celery workers (redis server required) ##
celery -A celery_worker.celery worker --loglevel=info -P solo

