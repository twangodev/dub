from taskiq_redis import ListQueueBroker, RedisAsyncResultBackend

from dub.config import settings

result_backend = RedisAsyncResultBackend(redis_url=settings.redis_url)
broker = ListQueueBroker(url=settings.redis_url).with_result_backend(result_backend)
