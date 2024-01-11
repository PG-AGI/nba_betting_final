# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class NbaBettingItem(scrapy.Item):
    game_id = scrapy.Field()
    game_datetime = scrapy.Field()
    home_team = scrapy.Field()
    away_team = scrapy.Field()
    home_score = scrapy.Field()
    away_score = scrapy.Field()
    open_line = scrapy.Field()
