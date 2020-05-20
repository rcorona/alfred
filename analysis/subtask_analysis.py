import pandas
import json

def process_entries(args, entries):
    df = pandas.DataFrame(entries)

    for groupby in args.groupby:
        #print(df.groupby(groupby)[args.metrics].mean())
        if ',' in groupby:
            groupby=list(groupby.split(','))
        gb = df.groupby(groupby)
        print(gb[args.metrics[0]].count())
        for key,row in gb[args.metrics].mean().iterrows():
            if isinstance(key,tuple):
                print(','.join(map(str,key)) + ',' + str(row[0]))
            else:
                print('{},{}'.format(key,row[0]))
        print()

def main(args):
    fname = args.subgoal_result_file
    with open(fname, 'r') as f:
        data = json.load(f)

    if 'successes' not in data:
        raise ValueError("json file {} does not contain a field successes".format(fname))
    
    if not isinstance(data['successes'], dict):
        raise ValueError("json file {}'s success field is not a dict; make sure it was produced by eval_subtasks (not eval_tasks)".format(fname))

    for partitions in [['successes'], ['failures'], ['successes', 'failures']]:
        partition_name = '+'.join(partitions)
        entries = [
            log_entry
            for partition in partitions
            for entries in data[partition].values()
            for log_entry in entries
        ]
        print("-" * 40)
        print("stats for {}".format(partition_name))
        process_entries(args, entries)
        print()
        print("-" * 40)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("subgoal_result_file")
    parser.add_argument("--metrics", nargs="+", default=['subgoal_success_spl'])
    parser.add_argument("--groupby", nargs="+", default=['subgoal_idx', 'subgoal_type'])
    args = parser.parse_args()
    main(args)
