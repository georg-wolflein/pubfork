import hydra
from omegaconf import OmegaConf


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
