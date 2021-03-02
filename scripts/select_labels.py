import argparse


def main(cluster_num):
    selected_clusters_test = []
    print("'y' for yes, 'n' for no, 'q' for quit, 'c' for change last one, default is no.")
    i = 0
    while i < cluster_num:
        user_input = input(f"Select cluster {i} ({len(selected_clusters_test)} of {cluster_num} selected) ? ")
        if user_input == 'n' or user_input == "":
            pass
        elif user_input == 'q':
            break
        elif user_input == 'c':
            i -= 1
            if selected_clusters_test[-1] == i:
                selected_clusters_test.pop()
            else:
                selected_clusters_test.append(i)
        else:
            selected_clusters_test.append(i)
        i += 1
    print(selected_clusters_test)
    selected_clusters_test_str = "--selected_clusters"
    for label in selected_clusters_test:
        selected_clusters_test_str += " " + str(label)
    print(f"{selected_clusters_test_str}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cluster_num', type=int, default=100)
    cfg = parser.parse_args()
    main(cfg.cluster_num)
