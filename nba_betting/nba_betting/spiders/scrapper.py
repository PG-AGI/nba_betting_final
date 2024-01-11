import scrapy
from datetime import datetime, timedelta
import logging

class NBASpider(scrapy.Spider):
    name = 'nba'
    allowed_domains = ['www.covers.com']
    start_urls = ['https://www.covers.com/sports/NBA/matchups']

    def start_requests(self):
        # Define the start and end dates for the date range you're interested in
        start_date = datetime(2019, 10, 12)
        end_date = datetime(2023, 11, 18)

        # Generate URLs with date parameters
        date_format = '%Y-%m-%d'
        date_range = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
        date_urls = [f'https://stats.nba.com/stats/leaguedashteamstats?selecteddate={date.strftime(date_format)}' for date in date_range]

        # Send requests for each URL in the date range
        for url in date_urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        # Extracting information about NBA teams
        matchups = response.css('.cmg_matchups_list .cmg_matchup_game_box.cmg_game_data')

        for matchup in matchups:
            game_id = matchup.css('::attr(data-game-id)').get()
            game_datetime_str = matchup.css('::attr(data-game-date)').get()
            home_team = matchup.css('::attr(data-home-team-shortname-search)').get()
            away_team = matchup.css('::attr(data-away-team-shortname-search)').get()
            home_score = matchup.css('::attr(data-home-score)').get()
            away_score = matchup.css('::attr(data-away-score)').get()
            open_line = matchup.css('::attr(data-game-odd)').get()

            # Convert date string to datetime object
            game_datetime = datetime.strptime(game_datetime_str, '%Y-%m-%d %H:%M:%S')

            yield {
                'game_id': game_id,
                'game_datetime': game_datetime,
                'home_team': home_team,
                'away_team': away_team,
                'home_score': home_score,
                'away_score': away_score,
                'open_line': open_line,
            }

        # Logging example
        self.log(f'Scraped data from {response.url}')

        # Follow pagination links and repeat the process
        next_page = response.css('a.next-page::attr(href)').get()
        if next_page is not None:
            yield response.follow(next_page, self.parse)

    def parse_errback(self, failure):
        # Log all failures
        self.logger.error(repr(failure))

        # Optionally handle specific exceptions
        if failure.check(scrapy.spidermiddlewares.httperror.HttpError):
            response = failure.value.response
            self.logger.error(f'HttpError on {response.url}')
        elif failure.check(scrapy.spidermiddlewares.offsite.OffsiteMiddleware):
            referer = failure.request.headers.get('Referer')
            self.logger.error(f'OffsiteMiddleware error. Referer: {referer}')
        elif failure.check(scrapy.spidermiddlewares.referer.RefererMiddleware):
            referer = failure.request.headers.get('Referer')
            self.logger.error(f'RefererMiddleware error. Referer: {referer}')