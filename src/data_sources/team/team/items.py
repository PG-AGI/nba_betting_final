import scrapy


class NbastatsGeneralTraditionalItem(scrapy.Item):
    team_name = scrapy.Field()
    to_date = scrapy.Field()
    season = scrapy.Field()
    season_type = scrapy.Field()
    games = scrapy.Field()
    gp = scrapy.Field()
    w = scrapy.Field()
    l = scrapy.Field()
    w_pct = scrapy.Field()
    min = scrapy.Field()
    fgm = scrapy.Field()
    fga = scrapy.Field()
    fg_pct = scrapy.Field()
    fg3m = scrapy.Field()
    fg3a = scrapy.Field()
    fg3_pct = scrapy.Field()
    ftm = scrapy.Field()
    fta = scrapy.Field()
    ft_pct = scrapy.Field()
    oreb = scrapy.Field()
    dreb = scrapy.Field()
    reb = scrapy.Field()
    ast = scrapy.Field()
    tov = scrapy.Field()
    stl = scrapy.Field()
    blk = scrapy.Field()
    blka = scrapy.Field()
    pf = scrapy.Field()
    pfd = scrapy.Field()
    pts = scrapy.Field()
    plus_minus = scrapy.Field()


class NbastatsGeneralAdvancedItem(scrapy.Item):
    team_name = scrapy.Field()
    to_date = scrapy.Field()
    season = scrapy.Field()
    season_type = scrapy.Field()
    games = scrapy.Field()
    gp = scrapy.Field()
    w = scrapy.Field()
    l = scrapy.Field()
    w_pct = scrapy.Field()
    min = scrapy.Field()
    e_off_rating = scrapy.Field()
    off_rating = scrapy.Field()
    e_def_rating = scrapy.Field()
    def_rating = scrapy.Field()
    e_net_rating = scrapy.Field()
    net_rating = scrapy.Field()
    ast_pct = scrapy.Field()
    ast_to = scrapy.Field()
    ast_ratio = scrapy.Field()
    oreb_pct = scrapy.Field()
    dreb_pct = scrapy.Field()
    reb_pct = scrapy.Field()
    tm_tov_pct = scrapy.Field()
    efg_pct = scrapy.Field()
    ts_pct = scrapy.Field()
    e_pace = scrapy.Field()
    pace = scrapy.Field()
    pace_per40 = scrapy.Field()
    poss = scrapy.Field()
    pie = scrapy.Field()


class NbastatsGeneralFourfactorsItem(scrapy.Item):
    team_name = scrapy.Field()
    to_date = scrapy.Field()
    season = scrapy.Field()
    season_type = scrapy.Field()
    games = scrapy.Field()
    gp = scrapy.Field()
    w = scrapy.Field()
    l = scrapy.Field()
    w_pct = scrapy.Field()
    min = scrapy.Field()
    efg_pct = scrapy.Field()
    fta_rate = scrapy.Field()
    tm_tov_pct = scrapy.Field()
    oreb_pct = scrapy.Field()
    opp_efg_pct = scrapy.Field()
    opp_fta_rate = scrapy.Field()
    opp_tov_pct = scrapy.Field()
    opp_oreb_pct = scrapy.Field()


class NbastatsGeneralOpponentItem(scrapy.Item):
    team_name = scrapy.Field()
    to_date = scrapy.Field()
    season = scrapy.Field()
    season_type = scrapy.Field()
    games = scrapy.Field()
    gp = scrapy.Field()
    w = scrapy.Field()
    l = scrapy.Field()
    w_pct = scrapy.Field()
    min = scrapy.Field()
    opp_fgm = scrapy.Field()
    opp_fga = scrapy.Field()
    opp_fg_pct = scrapy.Field()
    opp_fg3m = scrapy.Field()
    opp_fg3a = scrapy.Field()
    opp_fg3_pct = scrapy.Field()
    opp_ftm = scrapy.Field()
    opp_fta = scrapy.Field()
    opp_ft_pct = scrapy.Field()
    opp_oreb = scrapy.Field()
    opp_dreb = scrapy.Field()
    opp_reb = scrapy.Field()
    opp_ast = scrapy.Field()
    opp_tov = scrapy.Field()
    opp_stl = scrapy.Field()
    opp_blk = scrapy.Field()
    opp_blka = scrapy.Field()
    opp_pf = scrapy.Field()
    opp_pfd = scrapy.Field()
    opp_pts = scrapy.Field()
    plus_minus = scrapy.Field()
