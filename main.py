import os
import wandb
import hydra
from fedbox.utils.training import set_seed
from omegaconf import DictConfig, OmegaConf
from utils.misc import instantiate_from_config

config_path = os.environ.get("CONFIG_PATH")

@hydra.main(config_path=config_path, config_name='default')
def main(args):
    set_seed(args.seed)
    print(OmegaConf.to_yaml(args))

    # setup wandb
    wandb.init(
        project=args.wandb.project,
        entity=args.wandb.entity,
        name=args.wandb.name,
        mode=args.wandb.mode,
        tags=['train']
    )

    runner = instantiate_from_config(args)
    runner.run()

if __name__ == "__main__":
    main()

