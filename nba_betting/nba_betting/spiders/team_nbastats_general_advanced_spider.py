import scrapy


class TeamNbastatsGeneralAdvancedSpiderSpider(scrapy.Spider):
    name = "team_nbastats_general_advanced_spider"
    allowed_domains = ["stats.nba.com"]
    start_urls = ["https://stats.nba.com/stats/leaguedashteamstats"]

    def parse(self, response):
        pass
