import argparse
import json

import pandas as pd

from service.minimal_curve import compute_trace, compute_circus, compute_subpoints, visualise


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute trace of a well by list of 'INC' - zenit, 'AZI' - azimut, 'MD' - length of path.")
    parser.add_argument("--input", default="source.csv",
                        help="path to csv file with source data. (default=%(default)s)")
    parser.add_argument("--output", default="output.csv", required=False,
                        help="path to output file with computed trace. (default=%(default)s)")
    parser.add_argument("--deg2rad", default=True, required=False,
                        help="True if all angles in source in degrees. False otherwise. It impact to 'min_angel'. (default=%(default)s)")
    parser.add_argument("--min_angel", default=0.01, required=False,
                        help="minimum angel-teta to do compute. If the teta less then it curve replaced linear segment. (default=%(default)s)")
    parser.add_argument("--dL", default=10, required=False,
                        help="Approximate distance between subpoints. (default=%(default)s)")
    parser.add_argument("--visualise", default=False, required=False,
                        help="If True the result will be showed by matplot. (default=%(default)s)")
    parser.add_argument("--show_tangents", default=False, required=False,
                        help="If True tangents will be showed. (default=%(default)s)")
    parser.add_argument("--show_focuses", default=False, required=False,
                        help="If True all focuses of curves will be showed. (default=%(default)s)")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    try:
        df = pd.read_csv(args.input)
        trace_df = compute_trace(df, **vars(args))
        df = compute_circus(df)
        approximated = compute_subpoints(df, **vars(args))
        # store result
        columns = ['MD', 'X', 'Y', 'Z']
        raw = trace_df[columns].to_numpy().tolist()
        appr = approximated[columns].to_numpy().tolist()
        if '.csv' in args.output:
            approximated[columns].to_csv('output.csv', index=False)
        else:
            output_data = {
                "wells": [
                    {
                        "well_name": args.input,
                        "survey_data": raw,
                        "trajectory": appr,
                        "metadata": {},
                        "summary": {}
                    }
                ]
            }
            with open(args.output, 'w') as out:
                json.dump(output_data, out)
                out.flush()
        #
        if args.visualise:
            visualise(approximated, **vars(args))
    except FileNotFoundError:
        print("Файл не найден.")
    except Exception as e:
        print(f"Ошибка : {e}")
        raise e
