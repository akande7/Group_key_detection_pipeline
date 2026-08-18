def rule_based_prediction(row):
    if row["cross_layer_inconsistency_count"] > 0:
        return 1       

    return 0           
