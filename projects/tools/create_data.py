import argparse
import data_converter.pano_converter as pano_converter

def pano_data_prep(
    root_path, info_prefix, dataset_name, out_dir
):
    pano_converter.create_pano_infos(root_path, info_prefix)

parser = argparse.ArgumentParser(description="Data converter arg parser")
parser.add_argument("dataset", metavar="PanoDataset", help="name of the dataset")
parser.add_argument(
    "--root-path",
    type=str,
    default="/",
    help="specify the root path of dataset",
)
parser.add_argument(
    "--out-dir",
    type=str,
    default="/",
    required=False,
    help="name of info pkl",
)
parser.add_argument("--extra-tag", type=str, default="pano")

args = parser.parse_args()

if __name__ == "__main__":
    if args.dataset == "pano":

        pano_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            dataset_name="PanoDataset",
            out_dir=args.out_dir
        )