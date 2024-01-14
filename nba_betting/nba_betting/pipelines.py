import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
sys.path.append('/app/nba_betting/nba_betting')
from models import Base, Game
from sqlalchemy.exc import IntegrityError
from scrapy.exceptions import DropItem
import hashlib
from datetime import datetime

sys.path.append('/app')
from config import team_name_mapper

class PostgreSQLPipeline:
    
    def __init__(self, database_url):
        self.database_url = database_url
        self.scraped_items = 0
        self.errors = {"processing": []}

    @classmethod
    def from_crawler(cls, crawler):
        return cls(database_url=crawler.settings.get('DATABASE_URL'))

    def open_spider(self, spider):
        engine = create_engine(self.database_url)
        Base.metadata.create_all(engine)
        self.Session = sessionmaker(bind=engine)

    def close_spider(self, spider):
        pass

    def process_item(self, item, spider):
        session = self.Session()

        self.scraped_items += 1
        try:
            # Convert the game_datetime to the desired YYYYMMDD format for creating game_id.
            if isinstance(item["game_datetime"], str):
                item["game_datetime"] = datetime.strptime(item["game_datetime"], "%Y-%m-%d %H:%M:%S")

            # Extract the date from game_datetime and format it as a string
            game_date = item["game_datetime"].strftime("%Y%m%d")

            # Run the home_team and away_team through the team_name_mapper function
            mapped_home_team = team_name_mapper(item["home_team"])
            mapped_away_team = team_name_mapper(item["away_team"])

            # Concatenate game_date, mapped_home_team, and mapped_away_team to create the game_id
            game_id = f"{game_date}{mapped_home_team}{mapped_away_team}"

            print(game_id)

            # Assign the generated game_id and the mapped team names to the item
            item["game_id"] = game_id
            item["home_team"] = mapped_home_team
            item["away_team"] = mapped_away_team

            # Append the item to the list of scraped data
            self.nba_data.append(item)

        except Exception as e:
            self.errors["processing"].append([e, item])


        # Check if the game is completed based on the current date
        current_date = datetime.now()
        if item['game_datetime'] < current_date:
            item['game_completed'] = True
        elif item['game_datetime'] > current_date:
            item['game_completed'] = False
        else:
            # Check if the game is finished or not (customize this condition based on your requirements)
            if item['home_score'] is not None and item['away_score'] is not None:
                item['game_completed'] = True
            else:
                item['game_completed'] = False


        game = Game(
            game_id=item['game_id'],
            game_datetime=item['game_datetime'],
            home_team=item['home_team'],
            away_team=item['away_team'],
            home_score=item['home_score'],
            away_score=item['away_score'],
            open_line=item['open_line'],
        )

        try:
            session.add(game)
            session.commit()
        except IntegrityError:
            session.rollback()
            raise DropItem(f"Item with game_id {item['game_id']} already exists in the database.")
        except Exception as e:
            session.rollback()
            raise DropItem(f"Failed to save item to database: {e}")
        finally:
            session.close()

        return item

    def generate_game_id(self, game_datetime, home_team, away_team):
        # Customize this function based on your requirements
        data = f"{game_datetime}-{home_team}-{away_team}"
        return hashlib.md5(data.encode()).hexdigest()

    