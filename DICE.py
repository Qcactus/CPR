import os
from recq.recommenders.dice import DICE
from recq.utils import Dataset
from recq.tools.parser import parse_args
from recq.tools.io import print_seperate_line

args = parse_args("dice")
print_seperate_line()
for key, value in vars(args).items():
    print(key + '=' + str(value))
print_seperate_line()

curr_dir = os.path.dirname(__file__)
if args.gpu_id is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

data_dir = os.path.join(curr_dir, "data", args.dataset)
model_dir = os.path.join(curr_dir, "output", "model")

dataset = Dataset(args, data_dir)
model = DICE(args, dataset)
model.fit(args, model_dir)
