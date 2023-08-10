def probability_calculator(bow):
    probability = {}
    total_probability = sum(bow[1].values())
    for item in bow[1].items():
        probability[item[0]] = item[1] / total_probability
    return probability

