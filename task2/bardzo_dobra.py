import random

from task2.dostateczna import read_data, regression_curve_before, compare_data, regression_curve_after, mean_imputation
import numpy as np
import math


def percent(data_frame):
    count = data_frame.count()  # count of data in each column
    total_rows = len(data_frame.index)
    return ((len(count) * total_rows - sum(count)) / total_rows) * 100


def generate_missing_data(data_frame, column, percentile=15):
    # generate a list indexes of fields with the data
    id_val_pairs_with_no_missing_data = filter(lambda x: not math.isnan(x[1]), enumerate(list(data_frame[column])))
    indexes_with_no_missing_data = [index for index, _ in id_val_pairs_with_no_missing_data]

    # repeat until condition satisfied
    while percent(data_frame) < percentile:
        # remove random element from list of indexes
        index = indexes_with_no_missing_data.pop(random.randrange(len(indexes_with_no_missing_data)))
        # remove element corresponding to the index from data frame
        data_frame.loc[index, column] = float('NaN')

    return data_frame


def single_run(data, x_column, y_column, per):
    generate_missing_data(data, y_column, per)
    model, r_sq = regression_curve_before(data, x_column, y_column, "Graph of linear regression before ({}%)"
                                          .format(per))
    filled = mean_imputation(data)
    desc = "Porównanie charakterystyki zbiorów z brakami i po imputacji 'mean imputation' dla braku {}%"\
        .format(per)
    compare_data(data, filled, x_column, y_column, description=desc)
    regression_curve_after(filled, model, r_sq, x_column, y_column, "Graph of linear regression after ({}%)"
                           .format(per))


if __name__ == '__main__':
    y_column = "TOEFL_Score"
    x_column = "GRE_Score"

    data = read_data()
    single_run(data, x_column, y_column, 15)
    single_run(data, x_column, y_column, 30)
    single_run(data, x_column, y_column, 45)