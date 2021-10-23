import os

# import wandb
from recq.recommenders.mult_vae import MultVAE
from recq.utils import Dataset
from recq.tools.parser import parse_args
from recq.tools.io import print_seperate_line

args = parse_args("mult_vae")
print_seperate_line()
for key, value in vars(args).items():
    print(key + "=" + str(value))
print_seperate_line()

# wandb.init(project="mult_vae_" + args.dataset, config=args)
# wandb.define_metric("epoch")
# wandb.define_metric("val/*", step_metric="epoch", summary="max")

curr_dir = os.path.dirname(__file__)

data_dir = os.path.join(curr_dir, "data", args.dataset)
model_dir = os.path.join(curr_dir, "output", "model")

dataset = Dataset(args, data_dir)
model = MultVAE(args, dataset)
model.fit(args, model_dir)
