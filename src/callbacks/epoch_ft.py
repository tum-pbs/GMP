import lightning as L

class FineTuneCallback(L.Callback):
    def __init__(self, unfreeze_at_epoch: int = 5, unfreeze_modules: list = None):
        super().__init__()
        self.unfreeze_at_epoch = unfreeze_at_epoch
        self.unfreeze_modules = unfreeze_modules or []

    def on_train_epoch_start(self, trainer, pl_module):
        # Initial Setup at Epoch 0
        if trainer.current_epoch == 0:
            # 1. Freeze everything first
            for name, param in pl_module.named_parameters():
                    param.requires_grad = False
                    print(f"Freezing parameter: {name}")
            
            # 2. Search for the modules by name from config
            found_any = False
            for name, module in pl_module.named_modules():
                if any(target in name for target in self.unfreeze_modules):
                    for param in module.parameters():
                        param.requires_grad = True
                        print(f"Unfreezing module (warmup config): {name}")
                    found_any = True
            
            # 3. Dynamically unfreeze keys that were trained from scratch (mismatched/missing)
            model = getattr(pl_module, "model", None)
            missing_params = set()
            if model is not None and hasattr(model, "missing_keys"):
                for k in model.missing_keys:
                    missing_params.add(f"model.{k}")
            
            for name, param in pl_module.named_parameters():
                if name in missing_params:
                    param.requires_grad = True
                    print(f"Unfreezing parameter (trained from scratch): {name}")
                    found_any = True
            
            trainable = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
            frozen = sum(p.numel() for p in pl_module.parameters() if not p.requires_grad)
            total = trainable + frozen

            if not found_any:
                print(f"WARNING: None of the modules {self.unfreeze_modules} were found!")
            else:
                print(
                    f"--- Epoch 0: Active trainable parameters: {trainable:,} ({trainable/1e6:.2f}M, {100*trainable/total:.1f}%) | "
                    f"Frozen parameters: {frozen:,} ({frozen/1e6:.2f}M, {100*frozen/total:.1f}%) | Total: {total:,} ({total/1e6:.2f}M) ---"
                )

        # Unfreeze everything at the threshold
        elif trainer.current_epoch == self.unfreeze_at_epoch:
            for name, param in pl_module.named_parameters():
                param.requires_grad = True
                print(f"Unfreezing parameter: {name}")
            total = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
            print(f"--- Epoch {self.unfreeze_at_epoch}: All layers unfrozen ({total:,} parameters active, 100%) ---")