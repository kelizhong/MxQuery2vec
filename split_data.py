# -*- coding: utf-8 -*-
import redis
import time
import glob, os
import codecs
import re
from enum import Enum
from itertools import islice
# from pybloom import ScalableBloomFilter
from appmetrics import metrics


# sbf = ScalableBloomFilter(mode=ScalableBloomFilter.SMALL_SET_GROWTH)


class KeywordActionType(Enum):
    KEYWORDSBYADDS = 1
    KEYWORDSBYSEARCHS = 2
    KEYWORDSBYPURCHASES = 3
    KEYWORDSBYCLICKS = 4


class AksisData(object):
    def __init__(self, pattern):
        self.r = redis.Redis("krs-asin-datanode-1.aka.corp.amazon.com", 16379)
        self.pattern = pattern
        pool = redis.ConnectionPool(host='krs-asin-datanode-1.aka.corp.amazon.com', port=16379)
        redis_ins = redis.Redis(connection_pool=pool)
        self.redis_pipe = redis_ins.pipeline()

    def split_aksis_data(self):
        with codecs.open('add', 'w', encoding='utf-8', errors='ignore') as add, codecs.open('search', 'w',
                                                                                            encoding='utf-8',
                                                                                            errors='ignore') as search, codecs.open(
                'purchase', 'w', encoding='utf-8', errors='ignore') as purchase, codecs.open('click', 'w',
                                                                                             encoding='utf-8',
                                                                                             errors='ignore') as click:
            num = 0
            for line, action_type, title in self.read_data():
                line = line.strip()
                num += 1
                if num % 100 == 0:
                    if num > 500:
                        break
                    print(num)
                try:
                    if action_type == '1':
                        add.write(line + '\t' + title + '\n')
                        add.flush()
                    if action_type == '2':
                        search.write(line + '\t' + title + '\n')
                        search.flush()
                    if action_type == '3':
                        purchase.write(line + '\t' + title + '\n')
                        purchase.flush()
                    if action_type == '4':
                        click.write(line + '\t' + title + '\n')
                        click.flush()
                except UnicodeDecodeError:
                    pass

    def read_data(self):
        files = glob.glob(self.pattern)
        meter = metrics.new_meter("meter_speed")
        for file in files:
            print("reading file {}".format(file))
            n = 10000
            with codecs.open(file, encoding='utf-8', errors='ignore') as f:
                for n_lines in iter(lambda: tuple(islice(f, n)), ()):
                    meter.notify(n)
                    print(meter.get())
                    asins = []
                    action_types = []
                    for line in n_lines:
                        items = re.split(r'\t+', line)
                        if len(items) == 6:
                            asins.append(items[1])
                            action_types.append(items[4])
                    titles = self.get_asin_title(1, asins)
                    for line, action_type, title in zip(n_lines, action_types, titles):
                        if title is not None:
                            yield (line, action_type, title)

                """
                for line in f:
                    #MarketplaceId Asin Keyword Score ActionType Date
                    #ActioType: 1-KeywordsByAdds, 2-KeywordsBySearches, 3-KeywordsByPurchases, 4-KeywordsByClicks
                    items = re.split(r'\t+', line)
                    if len(items) == 6:
                        title = self.get_asin_title(items[0], items[1])
                        if title is not None:
                            print(items[1], title)
                """

    def get_asin_title(self, marketplaceId, asins):
        titles = []

        for asin in asins:
            key = '{}_{}'.format(marketplaceId, asin)
            self.redis_pipe.hmget(key, 'item_name')
        for h in self.redis_pipe.execute():
            titles.append(h[0])

        return titles


if __name__ == '__main__':
    pattern = './data/part*'
    a = AksisData(pattern)
    a.split_aksis_data()
