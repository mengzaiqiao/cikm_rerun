import sys
sys.path.append("../")

from beta_rec.datasets.atoswl import AtosWl


if __name__ == "__main__":
    dataset = AtosWl()
    dataset.preprocess()
    interactions = dataset.load_interaction()
    print(interactions.head())

#     # use filter_user_item_order() to make the dataset smaller, just for test
#     interactions = filter_user_item_order(interactions, 1, 1, 1)

#     # use n_test=1 to save time as well
#     dataset.make_temporal_basket_split(interactions)

#     dataset.load_leave_one_basket(n_test=1)
#     dataset.load_leave_one_out(n_test=1)
#     dataset.load_random_basket_split(n_test=1)
#     dataset.load_random_split(n_test=1)
#     dataset.load_temporal_basket_split(n_test=1)
#     train_data, valid_data_li, test_data_li = dataset.load_temporal_split(n_test=1)
