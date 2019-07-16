#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as pch
import random
import time
import matplotlib.cm as cm


# In[ ]:


homedir = os.path.expanduser("~")


# In[ ]:


dataFolder = homedir+"\\Documents\\GitHub\\open-data\\data\\"
matchesFolder = dataFolder + "matches\\"
lineupsFolder = dataFolder + "lineups\\"
eventsFolder = dataFolder + "events\\"


# In[ ]:


def json_loads(filepath):
    with open(filepath, encoding="utf-8") as json_file:
        n = json.load(json_file)
    return n


# In[ ]:


def tts(timestamp):
    n = 0
    for i in range(3):
        n += float(timestamp.split(":")[i]) * (60 ** (2-i))
    return n

def ttm(timestamp):
    n = tts(timestamp) / 60
    return n


# In[ ]:


def per90(DataFrame, columns):
    mins = "minutes_played"
    if mins in DataFrame.columns:
        m = DataFrame[mins]
    else:
        m = DataFrame["minutes_max"]
    n = pd.DataFrame(index = DataFrame.index)
    for col in columns:
        n[col] = 90*DataFrame[col]/m
    return n


# In[ ]:


def xgchain(matchDataFrame):
    aggFunc = {"shot_statsbomb_xg": "sum", "player_name": "unique"}
    df = matchDataFrame.groupby(["team_name", "possession"]).agg(aggFunc)
    Y = []
    for i, r in df.iterrows():
        for player in r.player_name:
            x = {"player_name": player, "team_name": i[0], "xg_chain": r.shot_statsbomb_xg}
            Y.append(x)
    ydf = pd.DataFrame(Y)
    new_df = ydf.groupby(["team_name", "player_name"]).xg_chain.sum()
    
    return new_df


# In[ ]:


def graph(figdim):
    fig, ax = plt.subplots(figsize=figdim, dpi = 150)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    return fig, ax, plt


# In[ ]:


def coordchanger(x, y, horizontal = True):
    if not horizontal:
        dim = y, x
    else:
        dim = x, y
    
    return dim


# In[ ]:


def draw_pitch(l = 12, linecolor="white", pitchcolor="seagreen", horizontal = True, half = False, attack = True):
    x, y = l, l*2/3
    
    if half:
        x = l/2
    dim = coordchanger(x, y, horizontal=horizontal)
    
    fig, ax, plt = graph((dim))
    fig.patch.set_facecolor(pitchcolor)
    
    x1, x2 = -5, 125
    y1, y2 = -5, 85
    
    if half:
        if attack:
            x1 = 60
        else:
            x2 = 60
        
    lim1 = coordchanger(x1, y1, horizontal=horizontal)
    lim2 = coordchanger(x2, y2, horizontal=horizontal)
    xlim = lim1[0], lim2[0]
    ylim = lim1[1], lim2[1]
    plt.xlim(xlim)
    plt.ylim(ylim)
    
    rect = [[[0, 0], [60, 80]],
            [[0, 30], [6, 20]],
           [[0, 18], [18, 44]],
           [[-2, 36], [2, 8]],
            [[60, 0], [60, 80]],
           [[114, 30], [6, 20]],
           [[102, 18], [18, 44]],
           [[120, 36], [2, 8]]]
    if half:
        if attack:
            rect = rect[len(rect)//2:]
        else:
            rect = rect[:len(rect)//2]
    for i in range(len(rect)):
        r = rect[i]
        start = list(coordchanger(r[0][0], r[0][1], horizontal=horizontal))
        width, height = coordchanger(r[1][0], r[1][1], horizontal=horizontal)
        patch = pch.Rectangle(start, height = height, width = width, fill = False, edgecolor = linecolor, linewidth = 2)
    
        ax.add_patch(patch)
    
    circ = [(12, 40), (108, 40)]
    if half:
        if attack:
            circ = [circ[1]]
        else:
            circ = [circ[0]]
 
    for i in range(len(circ)):
        c = coordchanger(circ[i][0], circ[i][1], horizontal = horizontal)
        patch = pch.Circle(c, 0.4, fill = True, facecolor = linecolor)
        
        ax.add_patch(patch)
        
    wedg = [[(60, 40), 10, 90, 270],
            [(60, 40), 0.4, 90, 270],
            [(12, 40), 10, 310, 50],
           [(60, 40), 10, 270, 90],
           [(60, 40), 0.4, 270, 90],
           [(108, 40), 10, 130, 230]]
    
    if half:
        if attack:
            wedg = wedg[3:]
        else:
            wedg = wedg[:3]
    
    for i in range(len(wedg)):
        w = wedg[i]
        if not horizontal:
            w[2] += 90
            w[3] += 90
        c = coordchanger(w[0][0], w[0][1], horizontal = horizontal)
        patch = pch.Wedge(center = tuple(c), r = w[1], theta1 = w[2]%360, theta2 = w[3]%360, color = linecolor, width = (0.4/w[1])**0.5)
        ax.add_patch(patch)
    
    plt.axis('off')
    return fig, ax, plt   


# In[ ]:


def titling(fig, ax, plt, c1, x, texty, deltay, t, h):
    for i in range(len(t)):
        if i == 0:
            w = "bold"
        else:
            w = "normal"
        plt.text(x, texty - i * deltay, t[i], ha=h, color=c1, fontsize=12, weight=w)
    return fig, ax, plt


# In[ ]:


def matchstep(matchDataFrame):
    DataFrame = matchDataFrame
    fig, ax, plt = graph((8, 6))
    plt.grid(True, axis="y", linestyle="--")

    plt.axhline(0, c = "k")
    plt.axvline(0, c = "k")
    shots_dfs = {}
    goals = {}
    for ha in ["home", "away"]:
        shots_dfs[ha] = DataFrame.loc[(((DataFrame.type_name == "Shot") & (DataFrame.team_name == DataFrame["_".join([ha, "team"])])) | (DataFrame.type_name.isin(["Half End", "Half Start"]))) & (DataFrame.period < 5), :]
        x = shots_dfs[ha].time.values
        y = shots_dfs[ha].shot_statsbomb_xg.fillna(0).cumsum()
        shots_dfs[ha] = shots_dfs[ha].assign(cumxg=y.values)
        ax.step(x=x, y=y.values, where="post", linewidth = 2)

        g = shots_dfs[ha].loc[shots_dfs[ha]["shot_outcome_name"] == "Goal"]
        og = DataFrame.loc[
            (DataFrame["type_name"] == "Own Goal Against") & (DataFrame.possession_team_name == DataFrame[ha + "_team"])]
        xa = g.time.values
        ya = g.cumxg.values

        for i in range(len(xa)):
            ax.annotate(g["player_name"].values[i], xy = (xa[i], ya[i]), xytext = (xa[i], ya[i]), ha = "right", fontsize = 8)
        if og["time"].count() > 0:
            xao = og.time.values
            for i in range(len(xao)):
                for j in range(len(x)):
                    if xao[i] < x[j]:
                        break
                ax.annotate(og["player_name"].values[i] + " (og)", xy=(xao[i], y.values[j-1]), xytext=(xao[i], y.values[j-1]),
                        ha = "right", fontsize=8)
        goals[ha] = g.time.count() + og.time.count()

    times = DataFrame.loc[DataFrame.type_name.isin(["Half End", "Half Start"]), "time"].unique()
    for i in range(1, min(DataFrame.period.max(), 4)+1):
        plt.axvline(x = times[i], c="seagreen", alpha=0.75)

    Y = np.concatenate((shots_dfs["home"].shot_statsbomb_xg.fillna(0).cumsum().values, shots_dfs["away"].shot_statsbomb_xg.fillna(0).cumsum().values))
    maxx = max(x)
    maxy = max(Y) + 0.5
    
    plt.xticks(np.arange(0, maxx + 1, 15))
    plt.yticks(np.arange(0, maxy, 0.5))
    ax.tick_params(axis ='both', which ='both', length = 0)
    plt.xlim(left=0, right=times[-1]+0.2) 
    plt.ylim(bottom=0)
    
    plt.xlabel("Match Time (minutes)")
    plt.ylabel("xG")

    maxy2 = plt.ylim()[1] * 1.2
    deltax = maxx/10
    deltay =  maxy2/25

    titling(fig, ax, plt, "C0", (maxx / 2) - deltax, maxy2, deltay, [DataFrame["home_team"].values[0], str(goals["home"]), str(shots_dfs["home"].loc[:, "shot_statsbomb_xg"].sum())[:4]], "right")
    titling(fig, ax, plt, "k", (maxx / 2), maxy2, deltay, [" v ", " Score ", "xG Total"], "center")
    titling(fig, ax, plt, "C1", (maxx / 2) + deltax, maxy2, deltay,
            [DataFrame["away_team"].values[0], str(goals["away"]), str(shots_dfs["away"].loc[:, "shot_statsbomb_xg"].sum())[:4]],
            "left")

    plt.tight_layout()
    return fig, ax, plt


# In[ ]:


def coordplot(x, y, horizontal=True, direction="right", half=False, attack = True):
    y = 80 - y
    if not horizontal:
        x, y = y, x
    if half:
        if not attack:
            x = 120 - x
    else:
        if direction == "left" or direction == "down":
            x = 120 - x
            y = 80 - y
            
    return x, y


# In[ ]:


def coordplot(x, y, horizontal=True, direction="right", half=False, attack = True):
    if direction == "right":
        y = 80 - y
        if half:
            if not attack:
                x = 120 - x
    elif direction == "up":
        x, y = y, x
        if half:
            if not attack:
                x = 120 - x
    elif direction == "down":
        x, y = 80 - y,  120 - x
        if half:
            if not attack:
                x = 120 - x
    else:
        x = 120 - x
        
    return x, y   


# In[ ]:


def matchloc(matchDataFrame):
    DataFrame = matchDataFrame
    fig, ax, plt = draw_pitch(linecolor="grey", pitchcolor="white")

    textx = sum(plt.xlim())/2
    deltax = textx/10
    ylim = plt.ylim()[1]
    
    if ylim > 100:
        texty = 135
    else:
        texty = 87
    deltay = texty/30
    text = "k"
    xg = "shot_statsbomb_xg"
    cmap = cm.get_cmap("Spectral")

    for team in DataFrame.possession_team_name.unique():
        shotdf = DataFrame.loc[(DataFrame.team_name == team) & (DataFrame.type_name == "Shot") & (DataFrame.shot_outcome_name != "Goal") & (DataFrame.period < 5)]
        goaldf = DataFrame.loc[(DataFrame.team_name == team) & (DataFrame.type_name == "Shot") & (DataFrame.shot_outcome_name == "Goal") & (DataFrame.period < 5)]
        ogdf = DataFrame.loc[(DataFrame.possession_team_name == team) & (DataFrame.type_name == "Own Goal Against")]
        for df in [shotdf, goaldf]:
            if team == shotdf.home_team.unique():
                direction = "right"
                col = "C1"
                piece = -1
                cmap = cm.get_cmap("Spectral")
            else:
                direction = "left"
                col = "C0"
                piece = 1
                cmap = cm.get_cmap("Spectral")
                
            x1 = np.array([n[0] for n in df.location.values])
            y1 = np.array([n[1] for n in df.location.values])

            x, y = coordplot(x1, y1, direction = direction)
            
            size = 100
            if df.shot_outcome_name.any() == "Goal":
                marker = "s"
            else:
                marker = "o"
            plt.scatter(x, y, s = size, edgecolors=text, linewidths=0.9, marker = marker, c = cmap(df[xg]/0.5), alpha= 0.5)

        titling(fig, ax, plt, col, textx + piece*deltax, texty, deltay, [team, len(goaldf)+len(ogdf), str(shotdf[xg].sum()+goaldf[xg].sum())[:4]], direction)

        
    central = [" v ", " Score ", " xG Total "]
    titling(fig, ax, plt, "black", textx, texty, deltay, central, "center")

    plt.tight_layout()
    return fig, ax, plt


# In[ ]:


def passmap(matchDataFrame):
    df = matchDataFrame.loc[(matchDataFrame.type_name == "Pass")&(matchDataFrame.pass_outcome_name.isna())]
    xy = ["x", "y"]
    cols = ["x_location", "y_location"]
    for i in xy:
        df[i + "_location"] = df.location.apply(lambda x: x[xy.index(i)])
        
    horiz = False

    gdf = df.groupby(["team_name", "player_name"])
    
    firstsub = df.loc[df.type_name == "Substitution", "time"]
    
    aggFunc = {"x_location": "mean", "y_location": "mean", "id": "count"}
    av = gdf.agg(aggFunc)
    av = av.merge(xgchain(matchDataFrame), on = ["team_name", "player_name"])
    
    rc = gdf.pass_recipient_name.value_counts()  
    teams = set(df.team_name)
    G = []
    for team in teams:
        team2 = list(teams - {team})[0]
        fig, ax, plt = draw_pitch(l = 10, linecolor="grey", pitchcolor="white", horizontal = horiz)
        x1 = av.loc[(team), "x_location"].values
        y1 = av.loc[(team), "y_location"].values

        x, y = coordplot(x1, y1, horizontal = horiz, direction = "up")
        
        s = np.array([1.5*n for n in av.loc[(team), "id"].fillna(0).values])
        labels = av.loc[team].index
        
        cmap1 = cm.get_cmap('BuPu')
        cmap2 = cm.get_cmap("Spectral")
        for i in range(len(rc)):
            v = rc.values[i]
            if rc.index[i][2] in av.loc[team].index and v > 0:
                x2 = av.loc[(team, rc.index.values[i][1])]["x_location"]
                y2 = av.loc[(team, rc.index.values[i][1])]["y_location"]
                
                xa, ya = coordplot(x2, y2, horizontal = horiz, direction = "up")
                
                x3 = av.loc[(team, rc.index.values[i][2])]["x_location"]
                y3 = av.loc[(team, rc.index.values[i][2])]["y_location"]
                
                xb, yb = coordplot(x3, y3, horizontal = horiz, direction = "up")
                
                dx = xb - xa
                dy = yb - ya
                
                c = cmap1(v/20)
                
                alpha = min([0.8, v/6])
                plt.arrow(xa, ya, dx, dy, width = 0.4, length_includes_head=True, shape = "right", ec = c, fc = c, alpha = alpha, ls = "-")
        
        xgc = av.loc[team].xg_chain.values
        plt.scatter(x, y, linewidths=s/2.5, s = s, marker = "o", c = "white", edgecolors = cmap2(xgc/2.5), alpha = 0.9)
        
        for i, label in enumerate(labels):
            ax.annotate(label.split(" ")[-1], xy = (x[i], y[i]), xytext = (x[i], y[i] - 0.9*np.log(s[i])), fontsize = 9, ha = "center")
                
        plt.title("{} Passmap v {}".format(team, team2))
        plt.tight_layout()
        G.append([(fig, ax, plt)])
    
    return G
    


# In[ ]:


def matchplots(eventsfilepath):
    DataFrame = matchdataframe(eventsfilepath)
    G = [[matchstep(DataFrame)], [matchloc(DataFrame)]] + passmap(DataFrame)
    home = DataFrame.home_team.unique()[0]
    away = DataFrame.away_team.unique()[0]

    return G, [home, away]


# In[ ]:


def PDO(DataFrame, addendum = "np_goal"):
    if "np" in addendum:
        x = "np_shot"
    else:
        s = addendum.split("_")[1]
        x = "shot_" + s
    pdo = 1000 * (DataFrame[addendum] / DataFrame[x] + (1 - DataFrame[addendum+"_opp"] / DataFrame[x + "_opp"]))
    return pdo


# In[ ]:


def matchdataframe(eventsfilepath):
    json = json_loads(eventsfilepath)
    matchlist = []
    halftimes = [0]

    for line in json:
        info = {"home_team": json[0]["team"]["name"], "away_team": json[1]["team"]["name"],
                "match_id": int(os.path.split(eventsfilepath)[-1].split(".")[0])}
        for r in line.keys():
            if isinstance(line[r], dict):
                for s in line[r].keys():
                    if isinstance(line[r][s], dict):
                        for t in line[r][s].keys():
                            info["_".join([r, s, t])] = line[r][s][t]
                    else:
                        info["_".join([r, s])] = line[r][s]
            else:
                info[r] = line[r]

        if line["type"]["name"] == "Half End":
            halftimes.append(ttm(line["timestamp"]))
        ht = halftimes[::2]
        info["time"] = ttm(line["timestamp"]) + sum(ht[:(info["period"])])
        matchlist.append(info)

    n = pd.DataFrame(matchlist)
    
    press = n.loc[n.type_name=="Pressure", :]
    time = press.time.values
    timex = time + 1/12

    for i in range(len(time)):
        reg = n.loc[(n.time >= time[i])&(n.time <= timex[i]), "possession_team_name"].nunique()-1
        n.loc[press.index[i], "pressure_regains"] = reg
    return n


# In[ ]:


def seasondataframe(matchesfilepath):
    season = json_loads(matchesfilepath)

    df_list = []
    for line in season:
        df = matchdataframe(eventsFolder + str(line["match_id"]) + ".json")
        df_list.append(df)
    bigdf = pd.concat(df_list, sort=True)

    for info in ["competition", "country", "season"]:
        if info == "country":
            z = "competition"
        else:
            z = info

        bigdf[info] = season[0][z][info+"_name"]

    bigdf.index = range(len(bigdf))

    return bigdf


# In[ ]:


def match_predict(MatchDataFrame):
    result = {"home": 0, "draw": 0, "away": 0}
    c = 500
    for j in range(c):
        G = []
        for team in [MatchDataFrame.home_team[0], MatchDataFrame.away_team[0]]:
            df = MatchDataFrame.loc[(MatchDataFrame.possession_team_name == team)&(MatchDataFrame.period < 5), :]
            g = df.loc[df.type_name == "Own Goal Against", "index"].count()
            for i in df.loc[df.shot_statsbomb_xg>0, "shot_statsbomb_xg"].values:
                r = random.random()
                if r < i:
                    g += 1
            G.append(g)
        if G[0] > G[1]:
            result["home"] += 100/c
        elif G[0] < G[1]:
            result["away"] += 100/c
        else:
            result["draw"] += 100/c

    return result


# In[ ]:


def general_rolling(x, y1, y2, labels, C = ["C0", "C1"]):
    fig, ax, plt = graph((8, 4))
    ax.axhline(y=0, c="k")
    plt.grid(True)

    plt.plot(x, y1, marker="o", label=labels[0], color=C[0])
    plt.plot(x, y2, marker="o", label=labels[1], color=C[1])

    plt.fill_between(x, y1, y2, where=y1 > y2, interpolate=True, alpha=0.5, color=C[0])
    plt.fill_between(x, y2, y1, where=y2 >= y1, interpolate=True, alpha=0.5, color=C[1])

    plt.legend(loc=4)

    return fig, ax, plt


# In[ ]:


def grouped_rolling(team, SeasonMatchDataFrame, roll = 5, colors = ["C0", "C1"]):
    DataFrame = SeasonMatchDataFrame
    df1 = DataFrame.loc[(slice(None), team), :]
    df2 = DataFrame.loc[DataFrame.opponent == team, :]
    
    if len(df1) > 2 * roll:
        x = range(1, len(df1) + 1)

        label = {}
        Y = {}

        n = 0
        Y[n] = df1.np_xg.rolling(roll).mean().values
        Y[n+1] = df2.np_xg.rolling(roll).mean().values
        label[n] = team
        label[n+1] = ["NPxGF", "NPxGA"]
        label[n+2] = ""

        n = 10
        Y[n] = df1.np_xg.rolling(roll).mean().values - df2.np_xg.rolling(roll).mean().values
        Y[n+1] = df1.np_goal.rolling(roll).mean().values - df2.np_goal.rolling(roll).mean().values
        label[n] = team
        label[n+1] = ["NPxGD", "NPGD"]
        label[n+2] = ""

        G = []
        for i in np.arange(0, n+1, 10):
            fig, ax, plt = general_rolling(x, Y[i], Y[i+1], labels=label[i+1], C=colors)

            sy = plt.ylim()[1]
            sx = sum(plt.xlim())/2

            plt.text(x = sx, y = sy*1.05, s="Rolling 5 Game Average", fontsize = 10, ha = "center")
            plt.text(x=sx, y=(sy+0.1) * 1.10, s=team, fontsize=12, ha="center")

            ax.axvline(x=0, c="k")

            plt.xticks(x[(roll-1)::roll])
            plt.xlabel("Matches played")
            plt.tight_layout()
            G.append([fig, ax, plt])
    else:
        G = team + " hasn't played enough games"

    return G


# In[ ]:


def general_scatter(x, y, labels, grid = True):
    fig, ax, plt = graph((8, 8))
    z = ((x-min(x))/(max(x) - min(x)) + (y - min(y))/(max(y) - min(y)))/2
    plt.scatter(x=x, y=y, marker = "o", edgecolors="k", alpha = 0.8, cmap="magma", c = z)

    x_delta = (max(x) - min(x))/50
    y_delta = (max(y) - min(y))/50
    for i, label in enumerate(labels):
        ax.annotate(label, xy = (x[i], y[i]), xytext = (x[i] + x_delta, y[i] - y_delta), fontsize = 7)

    return fig, ax, plt


# In[ ]:


def grouped_scatter(SeasonDataFrame, com):
    DataFrame = SeasonDataFrame
    a = com["competition_name"]
    b = com["season_name"]
    points = {"x": {}, "y": {}}
    labels = {"x_label": {}, "y_label": {}, "title": {}, "subtitle": {" ".join([a, b])}}

    n = 0
    points["x"][n] = (DataFrame["xg_open play"] - DataFrame["xg_open play_opp"])/DataFrame["matches"]
    points["y"][n] = PDO(DataFrame, "shot_open play_goal") - PDO(DataFrame, "xg_open play")
    labels["x_label"][n] = "Open Play xG difference per game"
    labels["y_label"][n] = "PDO - xPDO"
    labels["title"][n] = "Good v Luck"

    n = 1
    points["x"][n] = DataFrame["xg_open play"]/ DataFrame["shot_open play"]
    points["y"][n] = DataFrame["shot_open play"]/ DataFrame["matches"]
    labels["x_label"][n] = "xG per Open Play Shot"
    labels["y_label"][n] = "Open Play Shots per game"
    labels["title"][n] = "Open Play Attack Profile"

    n = 2
    points["x"][n] = DataFrame["xg_open play_opp"]/ DataFrame["shot_open play_opp"]
    points["y"][n] = DataFrame["shot_open play_opp"]/ DataFrame["matches"]
    labels["x_label"][n] = "xG per Open Play Shot against"
    labels["y_label"][n] = "Open Play Shots against per game"
    labels["title"][n] = "Open Play Defence Profile"

    n = 3
    points["x"][n] = DataFrame["xg_open play_opp"]/ DataFrame["matches"]
    points["y"][n] = DataFrame["xg_open play"]/ DataFrame["matches"]
    labels["x_label"][n] = "Open Play xG against per game"
    labels["y_label"][n] = "Open Play xG per game"
    labels["title"][n] = "Attack v Defence"

    teams = DataFrame.team_name

    G = []
    for i in range(n + 1):
        x = points["x"][i]
        y = points["y"][i]
        fig, ax, plt = general_scatter(x, y, teams)
        plt.grid(True)
        if min(y):
            ax.axhline(y=0, c="k")
        if min(x):
            ax.axvline(x=0, c="k")

        plt.xlabel(labels["x_label"][i])
        plt.ylabel(labels["y_label"][i])

        sy = plt.ylim()[1]
        sx = sum(plt.xlim())/2

        plt.text(x = sx, y = sy*1.05, s=labels["subtitle"], fontsize = 10, ha = "center")
        plt.text(x=sx, y=sy * 1.10, s=labels["title"][i], fontsize=12, ha="center")
        plt.tight_layout()

        G.append([fig, ax, plt])

    return G


# In[ ]:


def barxgchart(fullSeasonDataFrame, com):
    fig, ax, plt = graph((10, 10))
    
    time = fullSeasonDataFrame.minutes_played.median()
    fsdf = fullSeasonDataFrame.loc[fullSeasonDataFrame.minutes_played > time, :]
    l = ["np_xg", "xa"]
    df = per90(fsdf, l)
    df.loc[:, "x"] = sum([df.loc[:, i] for i in l])
    n = df.sort_values("x", ascending = False)[:30]

    x1 = n.np_xg.values
    x2 = n.xa.values
    y = n.index
    y_pos = np.arange(len(n), 0, -1)

    h = 0.5
    plt.barh(y_pos, x1, height = h, color = "C0")
    plt.barh(y_pos, x2, height = h, color = "C1", left = x1, tick_label = y)
    
    a = "NPxG + xA per 90:"
    b = com["competition_name"]
    c = com["season_name"]
    s = " ".join([a, b, c])
    plt.text(x = sum(plt.xlim())/2, y = len(n)+3, s = s, fontsize=12, ha="center")
    plt.text(x = sum(plt.xlim())/2, y = len(n)+2, s = "Minimum {} minutes".format(round(time, 1)), fontsize=10, ha="center")

    plt.grid(True)
    plt.tight_layout()
    return fig, ax, plt


# In[ ]:


def season_matches_summary(DataFrame, subject):
    grouping = ["match_id", "team_name"]
    if subject == "player":
        grouping += ["player_name"]
        
    DataFrame = DataFrame.loc[DataFrame.period < 5, :]

    df_list = []
    df = DataFrame.groupby(grouping)

    listing = ["shot_", "pass_", "duel_"]
    for x in listing + [""]:
        x_name = x + "type_name"
        df1 = df[x_name].value_counts().unstack().add_prefix(x)

        df_list += [df1]
        if x != "":
            group2 = grouping + [x + "type_name"]
            d2 = DataFrame.groupby(group2)
            df2 = d2[x + "outcome_name"].value_counts().unstack().unstack()
            if x == "shot_":
                df3 = d2.shot_statsbomb_xg.sum().unstack().add_prefix("xg_")
                df_list += [df3]
            df2.columns = ["_".join([x[:-1], y[1], y[0]]) for y in df2.columns.values]
            df_list += [df2]

    for x in ["dribble_outcome", "foul_committed_card", "bad_behaviour_card"]:
        prfx = x.split("_")[0]
        if x + "_name" in df.sum().columns:
            df1 = df[x+"_name"].value_counts().unstack().add_prefix(prfx+"_")
            df_list += [df1]
    
    df1 = df.pressure_regains.sum()
    df_list += [df1]
    
    if subject == "player":
        df1 = df.apply(lambda x: x[x["type_name"]=="Pass"].id.values)
        df2 = df1.apply(lambda x: DataFrame.loc[DataFrame["shot_key_pass_id"].isin(x), "shot_statsbomb_xg"].agg(["sum","count"]))
        df3 = df1.apply(lambda x: DataFrame.loc[(DataFrame["shot_outcome_name"]=="Goal")&(DataFrame["shot_key_pass_id"].isin(x)), "shot_statsbomb_xg"].count())

        df_list += [df2, df3]

    DF = pd.concat(df_list, join="outer", sort = True, axis= 1)

    DF = DF.rename(columns = {"sum": "xa", "count": "key_pass_total", 0: "assist_total"})
    DF = DF.fillna(0)
    
    goals_cols = [n for n in DF.columns if "Goal" in n and "shot" in n] + ["Own Goal Against"]
    DF.loc[:, "goal_total"] = DF[goals_cols].sum(axis=1)
    xg_cols = [n for n in DF.columns if "xg" in n]
    DF.loc[:, "xg_total"] = DF.loc[:, xg_cols].sum(axis=1)
    DF["shot_total"] = DF["Shot"]
    
    for match in DataFrame.match_id.unique():
        mdf = DataFrame.loc[DataFrame.match_id == match, :]
        DF.loc[match, "minutes_max"] = mdf.time.max()
        x1 = mdf.possession.max()
        X = pd.DataFrame(index = mdf.possession_team_name.unique())
        X["possession_count"] = mdf.groupby("possession_team_name").possession.nunique()
        X["possession_percent"] = X["possession_count"]/x1
    
    DF["matches"] = 1
    
    if subject == "team":
        DF["opponent"] = [DF.index[n + (-1)**n][1] for n in range(len(DF))]
        
    else:
        DF["goal_total"] -= DF["Own Goal Against"]
        DF["minutes_end"] = DF["minutes_max"]
        DF["appearance"] = 1
        DF["start"] = 1
        DF["minutes_start"] = (DataFrame.groupby(["match_id", "team_name", "substitution_replacement_name"]).time.min())
        DF = DF.fillna(0)
        DF.loc[DF.minutes_start>0, "start"] = 0
        DF["minutes_played"] = DF["minutes_end"] - DF["minutes_start"]
    
    for i in ["shot", "goal", "xg"]:
        if i == "goal":
            sj = "shot_Penalty_Goal"
        else:
            sj = i + "_Penalty"
        DF["NP_" + i] = DF[i + "_total"] - DF[sj]
        
    DF.columns = [n.lower() for n in DF.columns]
    return DF


# In[ ]:


def season_summary(smDataFrame):
    DF = smDataFrame
    season = DF.reset_index()
    if "start" in DF.columns:
        DF = season.groupby(["team_name", "player_name"]).sum()
    else:
        DF = season.groupby("team_name").sum()
        DF1 = season.groupby("opponent").sum()
        DF = DF.merge(DF1, on = DF.index, suffixes = ["", "_opp"]).rename(columns = {"key_0": "team_name"})
    return DF


# In[ ]:


def pressuremap(DataFrame, **kwargs):
    match_id_key = "match_id"
    grouping = [match_id_key, "team_name"]
    player = "player"
    a = (DataFrame.type_name == "Pressure")
    if kwargs is not None:
        for key in kwargs.keys():
            if key in ["player", "team"]:
                b = a & (DataFrame[key + "_name"] == kwargs[key])
        
        if match_id_key in kwargs.keys():
            c = b & (DataFrame[match_id_key] == kwargs[match_id_key])
        else:
            c = b
        
        
        if player in kwargs.keys():
            grouping += [player + "_name"]
              
    df = DataFrame.loc[c, :]
    dfloc = df.location
    
    horiz = True
    fig, ax, plt = draw_pitch(l = 10, linecolor="grey", pitchcolor="white", horizontal = horiz)
    
    x = np.array([n[0] for n in dfloc])
    y = np.array([n[1] for n in dfloc])

    xa, ya = coordplot(x, y, horizontal = horiz, direction = "right")
    
    binm = 10
    binn = 6
    bins = [binm, binn]
    if not horiz:
        bins[0], bins[1] = bins[1], bins[0]
        
    r = [[0, 120], [0, 80]]
    mdf = DataFrame.groupby("match_id")
    matches = mdf.time.count()
    minutes = mdf.time.max().sum()
    avg = DataFrame.loc[a, :]
    avgx = np.array([n[0] for n in avg.location])
    avgy = np.array([n[1] for n in avg.location])
    h, xedge, yedge = np.histogram2d(x=avgx, y=avgy, bins = bins, range = r)
    
    h1, xedges1, yedges1 = np.histogram2d(xa, ya, bins= bins, range = r)
    minutes1 = df.groupby("match_id").time.max().sum()
    
    mx = 0.1 * ((24/(binm * binn)) ** 0.5)
    data = h1.T/minutes1 - h.T/(minutes*2)
    plt.imshow(data, cmap = "Spectral", alpha = 0.7, vmin = -mx, vmax = mx, extent=[xedges1[0], xedges1[-1], yedges1[0], yedges1[-1]])
    plt.colorbar()
    
    marker = ["o", "s"]
    for i in range(2):
        reg = df.loc[df.pressure_regains == i, "location"]
        x = np.array([n[0] for n in reg])
        y = np.array([n[1] for n in reg])
        
        xa, ya = coordplot(x, y, horizontal = horiz, direction = "right")
        
        plt.scatter(xa, ya, c = "white", edgecolor = "k", marker = marker[i], s = 6, linewidths = 0.35)
        
    if "player" in kwargs.keys():
        title = kwargs["player"]
    elif "team" in kwargs.keys():
        title = kwargs["team"]
    
    plt.title(title + " Pressure Events & Heatmap")
    
    plt.tight_layout()
    return fig, ax, plt


# In[ ]:


comps = json_loads(dataFolder + "competitions.json")


# In[ ]:


LICT = []
for com in comps:
    start = time.time()
    DICT = {"comp_id": com["competition_id"], "season_id": com["season_id"]}
    fp = matchesFolder + "{}\\{}.json".format(com["competition_id"], com["season_id"])
    print(fp)
    d = seasondataframe(fp)
    DICT["seasondataframe"] = d
    for on in ["player", "team"]:
        df = season_matches_summary(d, on)
        DICT[on+"sms"] = df
        DICT[on+"sesu"] = season_summary(df)

    LICT.append(DICT)
    print(str(time.time() - start)[:5])


# In[ ]:


for com in comps:
    for line in LICT:
        if line["comp_id"] == com["competition_id"] and line["season_id"] == com["season_id"]:
            fig, ax, plt = barxgchart(line["playersesu"], com)
            plt.show()
            
            z = grouped_scatter(line["teamsesu"], com)
            for i in range(len(z)):
                z[i][2].show()
                
            for team in line["teamsesu"].team_name.values:
                z = grouped_rolling(team, line["teamsms"])
                if team in z:
                    print(z)
                else:
                    for i in range(len(z)):
                        z[i][2].show()


# In[ ]:


for com in comps:
    matches = json_loads(matchesFolder + "{}\\{}.json".format(com["competition_id"], com["season_id"]))
    for match in matches:
        start = time.time()
        a, teams = matchplots(eventsFolder + "{}.json".format(match["match_id"]))
        print(teams)
        for i in range(len(a)):
            a[i][0][2].show()
        print(time.time() - start)            


# In[ ]:


df = matchdataframe(eventsFolder + "{}.json".format(22984))
for line in LICT:
    if line["comp_id"]:
        df = line["seasondataframe"]
        for team in df.team_name.unique():
            fig, ax, plt = pressuremap(df, team=team)
            plt.show()


# In[ ]:


def shotheatmap(DataFrame, team):
    horiz = False
    attack = True
    half = True
    direction = "up"
    fig, ax, plt = draw_pitch(l = 10, linecolor="grey", pitchcolor="white", half = half, attack = attack, horizontal = horiz)
    
    cmap = cm.get_cmap("Spectral")
    filt = (DataFrame.type_name == "Shot") & (DataFrame.shot_type_name != "Penalty")
    totaldf = DataFrame.loc[filt, :]
    teamdf = DataFrame.loc[filt&(DataFrame.team_name == team)]
    
    xa = np.array([n[0] for n in teamdf.location])
    ya = np.array([n[1] for n in teamdf.location])
    c = teamdf.shot_statsbomb_xg.values
    
    x, y = coordplot(xa, ya, direction = "up", attack = attack, horizontal = horiz, half = half)
    
    
    bins = 8, 8
    r = [[60, 120], [0, 80]]

    X = []
    for df in [totaldf, teamdf]:
        mdf = df.groupby("match_id")
        matches = len(mdf.time.count())
        minutes = mdf.time.max().sum()
        
        df.x = np.array([n[0] for n in df.location])
        df.y = np.array([n[1] for n in df.location])
        
        s, xedge, yedge = np.histogram2d(x=df.x, y=df.y, bins=bins, range=r)
        
        X.append(s/minutes*90)
    
    data = X[1] - X[0]/2
    mx = 2*5/(bins[0] * bins[1])**0.5
    plt.imshow(np.rot90(data.T, 1), cmap = "Spectral", alpha = 0.4, vmin = -mx, vmax = mx, extent=[r[1][0],r[1][1],r[0][0],r[0][1]])
    plt.colorbar(shrink = 0.7)
    
    plt.scatter(x, y, s = 10, edgecolors="grey", linewidths=0.9, marker = "o", c = "white", alpha= 0.9)
    
    plt.title(team + " Shot Location Heatmap")
    plt.tight_layout()
    return fig, ax, plt


# In[ ]:


for line in LICT:
    if line["comp_id"]:
        df = line["seasondataframe"]
        for team in df.team_name.unique():
            fig, ax, plt = shotheatmap(df, team=team)
            plt.show()
            

