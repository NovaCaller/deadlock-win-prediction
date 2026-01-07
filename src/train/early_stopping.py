class EarlyStopping:
    def __init__(self, patience, min_improvement_pct):
        self.patience = patience
        self.min_improvement_pct = min_improvement_pct
        self.best_loss = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            return

        # check relative improvement
        improvement_threshold = self.best_loss * (1 - self.min_improvement_pct)

        if val_loss <= improvement_threshold:
            # significant improvement
            self.best_loss = val_loss
            self.best_model_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            self.counter = 0
        else:
            # not enough improvement
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)
        model.to(next(model.parameters()).device)