import random
import numpy as np
import torch
import os
import warnings
import hydra
import pickle
# import jpype
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict

# pytorch lightning
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger

# data loading
from src.data.dataset_block import get_dataset

# causal discovery
from src.causal_discovery.causal_discovery_block import causal_discovery

# graph completion block
from src.completion.completion_block import complete_graph_with_llm

# training and utils
from src.trainer import Trainer
from src.hydra_parsing import parse_hyperparams
from src.data.utils import static_graph_collate
from src.metrics import hamming_distance
from src.plots import maybe_plot_graph
from src.utils import get_intervention_policy, remove_cycles, remove_problematic_edges, maybe_corrupt_graph
from src.utils import clean_empty_configs, update_config_from_data, maybe_update_config_with_graph
from src.utils import finetune_model

# Suppress specific warning
warnings.filterwarnings("ignore", message="When grouping with a length-1 list-like")
    
def seed_everything(seed: int):
    print(f"Seed set to {seed}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@hydra.main(config_path="conf", config_name="my_sweep", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # various preliminaries, it set the seed for reproducibility
    torch.set_num_threads(cfg.get("num_threads", 1))
    seed_everything(cfg.get("seed"))
    os.mkdir('results')
    with open_dict(cfg): cfg.update(device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {cfg.device} device")

    # adjust config
    cfg = clean_empty_configs(cfg)

    # instantiate the dataset, split into train, val, test
    # preprocess all of them and save the preprocessed dataset
    dataset, true_graph, dataset_directory = get_dataset(cfg)

    # get the causal graph
    if cfg.dataset.load_true_graph:
        graph = true_graph
    else:
        if cfg.dataset.load_graph:
            with open(os.path.join(dataset_directory, "graph.pkl"), 'rb') as f:
                graph = pickle.load(f)
        else:
            # estimate causal graph with causal structural learning algorithms
            predicted_graph = causal_discovery(cfg, dataset, true_graph)
            if true_graph is not None:
                hamming = hamming_distance(true_graph, predicted_graph)
                print('(after CD) structural hamming distance: ', hamming)    

            # complete the causal graph with LLM and RAG
            completed_graph = complete_graph_with_llm(cfg, predicted_graph, cfg.dataset.name)
            if true_graph is not None:
                hamming = hamming_distance(true_graph, completed_graph)
                print('(after LLM + RAG) structural hamming distance: ', hamming)
            graph = completed_graph

            # save graph
            with open(os.path.join(dataset_directory, "graph.pkl"), 'wb') as f:
                pickle.dump(graph, f)

    # fix the graph
    # (part 1): remove bidirected and undirected edges + add virtual nodes
    # edge can only be directed at this stage, the following function is just here in 
    # case the CD + LLM + RAG pipeline is modified and could produce bidirected or undirected edges
    graph, dataset = remove_problematic_edges(graph, dataset)
    y_index = list(graph.index).index(dataset.y_info['names'][0]); assert y_index == len(graph) - 1
    # (part 2): remove cycles
    graph = remove_cycles(graph, y_index)

    if true_graph is not None:
        hamming = hamming_distance(true_graph, graph)
        print('(after fix) structural hamming distance: ', hamming)
    maybe_plot_graph(graph, 'fixed_graph')

    # corrupt the graph for ablation studies
    if cfg.dataset.corrupt_graph:
        corrupted_graph = maybe_corrupt_graph(graph, 
                                              corruption_type = cfg.dataset.corrupt_graph_type, 
                                              corrupt_graph_percentage = cfg.dataset.corrupt_graph_percentage)
        print('corrupted graph:', corrupted_graph)
        graph = corrupted_graph
        maybe_plot_graph(graph, 'corrupted_graph')

    # use the graph to define an intervention policy at test time
    interv_policy, ip_names = get_intervention_policy(cfg.policy, graph, true_graph, y_index)
    print('intervention policy:', interv_policy)
    print('intervention policy names:', ip_names)

    # update config based on the dataset
    # e.g., set input and output size of the model
    cfg = update_config_from_data(cfg, dataset)
    cfg = maybe_update_config_with_graph(cfg, graph, interv_policy)
    
    ############ model block ########################################################################################
    [dataset.data[split].register_graph(graph) for split in dataset.data]
    train_dataloader = DataLoader(dataset.data['train'], 
                                  batch_size=cfg.dataset.batch_size, 
                                  collate_fn=static_graph_collate,
                                  num_workers=cfg.dataset.num_workers)
    val_dataloader = DataLoader(dataset.data['val'], 
                                batch_size=cfg.dataset.batch_size, 
                                collate_fn=static_graph_collate,
                                num_workers=cfg.dataset.num_workers)
    test_dataloader = DataLoader(dataset.data['test'], 
                                 batch_size=cfg.dataset.batch_size, 
                                 collate_fn=static_graph_collate,
                                 num_workers=cfg.dataset.num_workers)
    
    engine = instantiate(cfg.engine)
    try:
        trainer = Trainer(cfg)
        trainer.logger.log_hyperparams(parse_hyperparams(cfg))
        # ---- train
        trainer.fit(engine, train_dataloader, val_dataloader)
        # ---- finetune the encoder (eventually)
        if cfg.dataset.loader.ftune_size > 0: 
            trainer, engine = finetune_model(cfg, engine, dataset)
        # ----- test
        trainer.test(engine, test_dataloader, ckpt_path='best')
        trainer.logger.finalize("success")
    finally:
        if isinstance(trainer.logger, WandbLogger):
            trainer.logger.experiment.finish()
    ############################################################################################


if __name__ == "__main__":
    main()
    print('done')