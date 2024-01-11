from sqlalchemy import Column, String, Integer, DateTime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Game(Base):
    __tablename__ = 'games'

    game_id = Column(String(255), primary_key=True)
    game_datetime = Column(DateTime)
    home_team = Column(String(255))
    away_team = Column(String(255))
    home_score = Column(Integer)
    away_score = Column(Integer)
    open_line = Column(String(255))
