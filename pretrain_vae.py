import argparse
import sys
from tqdm import tqdm
import pdb

def train_for_folder(
    root_dir="CrysLDNet",
    config_name="config.json",
    classification_threshold=None,
    batch_size=None,
    start_epochs=None,
    epochs=None,
    restart_model_path=None,
    file_format="poscar",
    output_dir=None,
    pre_train = False,
    mask_ratio=None,
    debug = False
):
    from jarvis.db.jsonutils import dumpjson
    from functools import partial
    import numpy as np
    import csv
    import os
    import time
    from jarvis.core.atoms import Atoms
    from data_accelerate import get_train_val_loaders
    from jarvis.db.jsonutils import loadjson
    import glob
    import torch
    from config import TrainingConfig
    
    
    from models.pddformer import eComformer
    
    from accelerate import Accelerator
    from acc_util import get_data, Criterion
    import json
    from torch.utils.data import DataLoader
    from accelerate.utils import gather_object,broadcast_object_list
    from train_accelerator import  group_decay, setup_optimizer, CustomMetric_atom, CustomMetric_position, CustomMetric_lattice
    import gc
    accelerator = Accelerator()
    
    device=accelerator.device

    if mask_ratio is not None:
        mask_ratio = float(mask_ratio)

    if output_dir is not None:
        output_dir = f'{output_dir}m{mask_ratio}/'
    """Train for a folder."""
    
    config = {
        "dataset": 'user_data',
        "target": 'target',
        "start_epochs": start_epochs,
        "epochs": epochs,  # 00,#00,
        "batch_size": batch_size,  # 0,
	"max_neighbors":25,
        "cutoff":4.0,
        "weight_decay": 1e-05,
        "learning_rate": 0.001,
        "criterion": "mse",
        "optimizer": "adamw",
        "scheduler": "onecycle",
        "save_dataloader": False,
        "pin_memory": False,
        "write_predictions": True,
        "num_workers": 0,
        "classification_threshold": classification_threshold,
        "atom_features": 'cgcnn',
        'pre_train': True,
        "pyg_input": True,
        "use_lattice": True,
        "use_angle": False,
        "output_dir": output_dir,
        "model": {
            "use_angle": False,
            "name": 'eComformer',
            "pre_train": True,
            "mask_ratio":mask_ratio,
            "output_features": 119,
        },
        "pre_train": True
    }

    pre_train = config["pre_train"]
    print(config)
    if type(config) is dict:
        try:
            # pdb.set_trace()
            config = TrainingConfig(**config)
        except Exception as exp:
            print("Check", exp)
            print('error in converting to training config!')
    _model = {
        "eComformer" : eComformer,
    }
    # print(restart_model_path)
    if restart_model_path is not None:
        print("Restarting model from:", restart_model_path)
        rest_config = loadjson(os.path.join(restart_model_path,"config.json"))
        print("rest_config", rest_config)
        rest_config_tt = TrainingConfig(**rest_config)
        model = Matformer(rest_config_tt.model)
        print("Checkpoint file", restart_model_path)
        model.load_state_dict(torch.load(os.path.join(restart_model_path,"best.pt")))#, map_location=device))
        model#.to(device)
    else:
        model = None

    if accelerator.is_local_main_process:
        # pdb.set_trace()
        print("*******train_for_folder***********")
        dataset, n_outputs = get_data(root_dir, file_format='poscar', debug=debug)
        
        (
            train_data, val_data, test_data, collate_fn, pin_memory
        ) = get_train_val_loaders(
            dataset_array=dataset,
            target=config.target,
            n_train=config.n_train,
            n_val=config.n_val,
            n_test=config.n_test,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
            batch_size=config.batch_size,
            atom_features=config.atom_features,
            neighbor_strategy=config.neighbor_strategy,
            standardize=config.atom_features != "cgcnn",
            id_tag=config.id_tag,
            pin_memory=config.pin_memory,
            workers=0,
            save_dataloader=config.save_dataloader,
            use_canonize=config.use_canonize,
            filename=config.filename,
            cutoff=config.cutoff,
            max_neighbors=config.max_neighbors,
            output_features=config.model.output_features,
            classification_threshold=config.classification_threshold,
            target_multiplication_factor=config.target_multiplication_factor,
            standard_scalar_and_pca=config.standard_scalar_and_pca,
            keep_data_order=config.keep_data_order,
            output_dir=config.output_dir,
            pre_train=pre_train,
            mask_ratio=mask_ratio,
        )
        gc.collect()
        if not os.path.exists(config.output_dir):
            os.makedirs(config.output_dir)
        tmp = config.dict()
        f = open(os.path.join(config.output_dir, "config.json"), "w")
        f.write(json.dumps(tmp, indent=4))
        f.close()
        train_data_list = [train_data]
        val_data_list = [val_data]
        test_data_list = [test_data]
        collate_fn_list = [collate_fn]
        pin_memory_list = [pin_memory]
        #assert not None in train_data_list
        #train_data = train_data_list[0]
        #val_data = val_data_list[0]
        #test_data = test_data_list[0]
        #collate_fn = collate_fn_list[0]
        #pin_memory = pin_memory_list[0]
        print("## Dataset constructed ##")
    else:
        train_data_list = [None]
        val_data_list = [None]
        test_data_list = [None]
        collate_fn_list = [None]
        pin_memory_list = [None]
        print("Waiting")
    accelerator.wait_for_everyone()
    print("## Start Broadcast ##")
    broadcast_object_list(train_data_list)
    broadcast_object_list(val_data_list)
    broadcast_object_list(test_data_list)
    broadcast_object_list(collate_fn_list)
    broadcast_object_list(pin_memory_list)
    print("## Broadcast Done ##")    
    if not accelerator.is_local_main_process:
        train_data = train_data_list[0]
        val_data = val_data_list[0]
        test_data = test_data_list[0]
        collate_fn = collate_fn_list[0]
        pin_memory = pin_memory_list[0]
    del train_data_list
    del val_data_list
    del test_data_list
    del collate_fn_list
    del pin_memory_list
    gc.collect()
    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    
    if config.save_dataloader and accelerator.is_local_main_process:
        train_sample = os.path.join(output_dir, config.filename + "_train.data")
        val_sample = os.path.join(output_dir, config.filename + "_val.data")
        test_sample = os.path.join(output_dir, config.filename + "_test.data")
        torch.save(train_loader, train_sample)
        torch.save(val_loader, val_sample)
        torch.save(test_loader, test_sample)
    
        print("n_train:", len(train_loader.dataset))
        print("n_val:", len(val_loader.dataset))
        print("n_test:", len(test_loader.dataset))
    
    _model = {
        "eComformer" : eComformer,
    }
    # pdb.set_trace()
    if restart_model_path is not None:
        net = model
    else:
        net = _model.get(config.model.name)(config.model)
    net.to(device)
    params = group_decay(net)
    optimizer = setup_optimizer(params, config)
    if config.scheduler == "none":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: 1.0
        )

    elif config.scheduler == "onecycle":
        steps_per_epoch = len(train_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            epochs=config.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,
        )
    elif config.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=100000,
            gamma=0.96,
    )    
    criterion = Criterion(config.mask_ratio)
    if accelerator.is_local_main_process:
        metrics = {"loss": criterion}
        history_dict_train = {}
        history_dict_val = {}
        history_dict_test = {}
        history_dict_train["loss"] = []
        history_dict_val["loss"] = []
        history_dict_test["loss"] = []
        if config.model.mask_ratio is not None:
            metrics["atom_acc"] = CustomMetric_atom()
            history_dict_train["atom_acc"] = []
            history_dict_val["atom_acc"] = []
            history_dict_test["atom_acc"] = []
        
        metrics["position_loss"] = CustomMetric_position()
        history_dict_train["position_loss"] = []
        history_dict_val["position_loss"] = []
        history_dict_test["position_loss"] = []
        
        metrics["lattice_loss"] = CustomMetric_lattice()
        history_dict_train["lattice_loss"] = []
        history_dict_val["lattice_loss"] = []
        history_dict_test["lattice_loss"] = []
        store_list = []
        for i in range(5):
            store = config.epochs-10*i
            if store > 0:
                store_list.append(store)
            else:
                break

    best_loss = np.inf
    net,optimizer,train_loader,val_loader,test_loader,scheduler = \
        accelerator.prepare(net,optimizer,train_loader,val_loader,test_loader,scheduler)

    if accelerator.is_local_main_process:
        t1 = time.time()
        print("********START TRAINING***********")
    for e in range(config.start_epochs,config.epochs):
        for inx, data in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            #pdb.set_trace()
            results = net([data[0], data[1]]) 
            loss = criterion.forward(results, data[2])
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
        with torch.no_grad():        
            if e%10 == 0:
                for train_data in train_loader:
                    return_dict = net((train_data[0], train_data[1])) # Matformer
                    # return_dict = net(train_data[0]) # PotNet
                    if accelerator.is_local_main_process:        
                        for kk in metrics:
                            metrics[kk].update((return_dict, train_data[2]))
                if accelerator.is_local_main_process:
                    for kk in metrics:
                        history_dict_train[kk].append(metrics[kk].compute())
                        metrics[kk].reset()
                optimizer.zero_grad()
                
            for eval_data in val_loader:
                return_dict = net((eval_data[0], eval_data[1])) # Matformer
                # return_dict = net(eval_data[0]) # PotNet
                if accelerator.is_local_main_process:
                    for kk in metrics:
                        metrics[kk].update((return_dict, eval_data[2]))
                    print_str = []
                optimizer.zero_grad()
            if accelerator.is_local_main_process:
                for kk in metrics:
                    history_dict_val[kk].append(metrics[kk].compute())
                    metrics[kk].reset()
                    print_str.append(kk)
                    print_str.append(str(history_dict_val[kk][-1]))

                # pdb.set_trace()
                my_string = ' '.join(print_str)
                print(f'{e+1}/{config.epochs}:{my_string}')
                if best_loss > history_dict_val["loss"][-1]:
                    best_loss = history_dict_val["loss"][-1]
                    unwrap_model=accelerator.unwrap_model(net)
                    torch.save(
                        unwrap_model.state_dict(),
                        os.path.join(config.output_dir, f'best.pt'),
                    )
                if e in store_list:
                    torch.save(
                        unwrap_model.state_dict(),
                        os.path.join(config.output_dir, f'model_{e}.pt'),
                    )
    
    for test_data in test_loader:
        optimizer.zero_grad()
        with torch.no_grad():
            return_dict = net([test_data[0], test_data[1]]) # Matformer
            # return_dict = net(eval_data[0]) # PotNet
            if accelerator.is_local_main_process:
                for kk in metrics:
                    metrics[kk].update([return_dict, test_data[2]])
                for kk in metrics:
                    history_dict_test[kk].append(metrics[kk].compute())

    if accelerator.is_local_main_process:
        t2 = time.time()
        print("Total time:", t2-t1)
        history_dict_test["time"] = t2-t1
        dumpjson(
            filename=os.path.join(config.output_dir, "history_dict_test.json"),
            data=history_dict_test,
        )
        dumpjson(
            filename=os.path.join(config.output_dir, "history_dict_val.json"),
            data=history_dict_val,
        )
        dumpjson(
            filename=os.path.join(config.output_dir, "history_dict_train.json"),
            data=history_dict_train,
        )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CrysLDNet")    
    parser.add_argument(
        "--root_dir",
        default="/home/partha/Shrimon_Workspace/CrystalLatent_comformer/pretraining_data_jarvis",
        help="Folder with id_props.csv, structure files",
    )
    parser.add_argument(
        "--config_name",
        default="CrysLDNet",
        help="Name of the config file",
    )

    parser.add_argument(
        "--file_format", default="poscar", help="poscar/cif/xyz/pdb file format."
    )
    parser.add_argument(
        "--classification_threshold",
        default=None,
        help="Floating point threshold for converting into 0/1 class"
        + ", use only for classification tasks",
    )

    parser.add_argument(
        "--batch_size", default=256, help="Batch size, generally 64"
    )

    parser.add_argument(
        "--start_epochs", default=0, help="Number of epochs, generally 300"
    )
    
    parser.add_argument(
        "--epochs", default=50, help="Number of epochs, generally 300"
    )

    parser.add_argument(
        "--output_dir",
        default="log/CrysLDNet_vae/",
        help="Folder to save outputs",
    )

    parser.add_argument(
        "--device",
        default=None,
        help="set device for training the model [e.g. cpu, cuda, cuda:2]",
    )

    parser.add_argument(
        "--restart_model_path",
        default=None,
        help="Checkpoint file path for model",
    )

    parser.add_argument(
        "--pre_train",
        default=True,
        help="pre_train or not",
    )

    parser.add_argument(
        "--mask_ratio",
        default=None,
        help="Mask ratio",
    )

    parser.add_argument(
        "--debug",
        default= False,
        help="debug or not",
    )
    args = parser.parse_args(sys.argv[1:])
    train_for_folder(
        root_dir = args.root_dir,
        config_name=args.config_name,
        classification_threshold=args.classification_threshold,
        output_dir=args.output_dir,
        batch_size=(args.batch_size),
        start_epochs=(args.start_epochs),
        epochs=(args.epochs),
        file_format=(args.file_format),
        restart_model_path=(args.restart_model_path),
        mask_ratio = (args.mask_ratio),
        debug = (args.debug)
    )
