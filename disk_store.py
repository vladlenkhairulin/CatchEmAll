"""Словарь на диске и сортировка больших файлов."""
import heapq
import json
from collections import OrderedDict
from contextlib import ExitStack
from pathlib import Path

from sqlitedict import SqliteDict


class DiskStore:
    def __init__(self, path, cache_size=30000):
        self.db = SqliteDict(str(path), autocommit=False, outer_stack=False)
        # Частые значения оставляем в памяти.
        self.cache = OrderedDict()
        self.cache_size = cache_size

    def get(self, category, key, default=None):
        address = category + "|" + key
        if address in self.cache:
            value = self.cache.pop(address)
        else:
            value = self.db.get(address)
        self.cache[address] = value
        if len(self.cache)>self.cache_size:
            self.cache.popitem(last=False)
        return default if value is None else value

    def put(self, category, key, value):
        address = category + "|" + key
        self.db[address] = value
        self.cache.pop(address, None)
        self.cache[address] = value
        if len(self.cache)>self.cache_size:
            self.cache.popitem(last=False)

    def commit(self):
        self.db.commit()

    def close(self):
        self.cache.clear()
        self.db.close()


def ordered_records(path, scratch, chunk_size=50000):
    """Сортируем по частям, потом собираем вместе."""
    scratch = Path(scratch)
    scratch.mkdir(exist_ok=True)
    chunks = []
    try:
        with Path(path).open(encoding="utf-8") as stream:
            finished = False
            while not finished:
                batch = []
                for _ in range(chunk_size):
                    line = stream.readline()
                    if not line:
                        finished = True
                        break
                    batch.append(json.loads(line))
                if not batch:
                    break
                batch.sort()
                chunk = scratch/f"chunk_{len(chunks):06d}.jsonl"
                with chunk.open("w",encoding="utf-8") as output:
                    for record in batch:
                        output.write(json.dumps(record,ensure_ascii=False)+"\n")
                chunks.append(chunk)
        with ExitStack() as stack:
            streams = [stack.enter_context(p.open(encoding="utf-8")) for p in chunks]
            records = [(json.loads(line) for line in stream) for stream in streams]
            yield from heapq.merge(*records)
    finally:
        for chunk in chunks:
            chunk.unlink(missing_ok=True)
