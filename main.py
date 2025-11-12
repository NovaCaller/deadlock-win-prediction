import requests

base_url = "https://api.deadlock-api.com"

def get_from_api(match_id):
    metadata_url = base_url + "/v1/matches/" + str(match_id) + "/metadata"
    response = requests.get(metadata_url)
    return response.json()

def calculate_net_income(json_data):
    net_incomes_team_1 = {}
    net_incomes_team_2 = {}

    json_data = json_data["match_info"]
    players = json_data["players"]

    for player in players:
        stats = player["stats"]
        team = player["team"]
        for stat in stats:
            timestamp = stat["time_stamp_s"]
            timestamp = int(timestamp)
            net_worth = stat["net_worth"]

            if team == 1:
                if timestamp not in net_incomes_team_1:
                    net_incomes_team_1[timestamp] = net_worth
                else:
                    net_incomes_team_1[timestamp] += net_worth
            else:
                if timestamp not in net_incomes_team_2:
                    net_incomes_team_2[timestamp] = net_worth
                else:
                    net_incomes_team_2[timestamp] += net_worth

    print("Team 1 : " + str(net_incomes_team_1))
    print("Team 2 :  " + str(net_incomes_team_2))
    return net_incomes_team_1, net_incomes_team_2


if __name__ == '__main__':
    calculate_net_income(get_from_api(45932614))