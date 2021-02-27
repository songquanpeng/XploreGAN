import argparse


def main(cluster_num):
    selected_labels = []
    print("'y' for yes, 'n' for no, 'q' for quit, 'c' for change last one, default is no.")
    i = 0
    while i < cluster_num:
        user_input = input(f"Select cluster {i} ({len(selected_labels)} of {cluster_num} selected) ? ")
        if user_input == 'n' or user_input == "":
            pass
        elif user_input == 'q':
            break
        elif user_input == 'c':
            i -= 1
            if selected_labels[-1] == i:
                selected_labels.pop()
            else:
                selected_labels.append(i)
        else:
            selected_labels.append(i)
        i += 1
    print(selected_labels)
    selected_labels_str = "--selected_clusters"
    for label in selected_labels:
        selected_labels_str += " " + str(label)
    print(f"{selected_labels_str}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cluster_num', type=int, default=100)
    cfg = parser.parse_args()
    main(cfg.cluster_num)
