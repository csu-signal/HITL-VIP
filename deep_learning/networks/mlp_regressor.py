import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics


class mlp_regressor(pl.LightningModule):
    def __init__(
        self,
        n_inputs: int,
        n_hiddens_per_layer: list[int],
        n_outputs: int,
        act_func: str = "tanh",
        Xmeans: torch.Tensor = None,
        Xstds: torch.Tensor = None,
        Tmeans: torch.Tensor = None,
        Tstds: torch.Tensor = None,
        optimizer: str = "adam",
        learning_rate: float = 1e-5,
    ):
        super(
            mlp_regressor, self
        ).__init__()  # call parent class (torch.nn.Module) constructor

        self.save_hyperparameters()

        # Set self.n_hiddens_per_layer to [] if argument is 0, [], or [0]

        if (
            n_hiddens_per_layer == 0
            or n_hiddens_per_layer == []
            or n_hiddens_per_layer == [0]
        ):
            self.n_hiddens_per_layer = []
        else:
            self.n_hiddens_per_layer = n_hiddens_per_layer

        self.Xmeans = Xmeans
        self.Xstds = Xstds
        self.Tmeans = Tmeans
        self.Tstds = Tstds

        self.hidden_layers = torch.nn.ModuleList()  # necessary for model.to('cuda')

        for nh in self.n_hiddens_per_layer:
            self.hidden_layers.append(
                torch.nn.Sequential(
                    torch.nn.Linear(n_inputs, nh),
                    torch.nn.Tanh() if act_func == "tanh" else torch.nn.ReLU(),
                )
            )

            n_inputs = nh

        self.output_layer = torch.nn.Linear(n_inputs, n_outputs)

        self.opt = optimizer
        self.lr = learning_rate
        self.criterion = torch.nn.HuberLoss()
        self.metric = "MAE"
        self.metric_function = torchmetrics.functional.mean_absolute_error
        self.error_trace = []
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, X):
        Y = X
        for hidden_layer in self.hidden_layers:
            Y = hidden_layer(Y)
        Y = self.output_layer(Y)
        return Y

    def use(self, X: torch.Tensor, device: torch.device = torch.device("cpu")):
        # Set input matrix to torch.tensors if not already.
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X).float()

        X = X.to(device)
        self.Xmeans = self.Xmeans.to(device)
        self.Xstds = self.Xstds.to(device)
        self.Tmeans = self.Tmeans.to(device)
        self.Tstds = self.Tstds.to(device)

        X = (X - self.Xmeans) / self.Xstds

        Y = self.forward(X)
        Y = (Y * self.Tstds) + self.Tmeans

        return Y.cpu().detach().numpy()

    def training_step(self, batch, batch_idx):
        X, T = batch

        out = self.forward(X)
        loss = self.criterion(out, T)
        metric_score = self.metric_function(
            out,
            T,
        )

        self.log(f"train_{self.metric}_batch", metric_score, prog_bar=True)
        self.training_step_outputs.append({"loss": loss, self.metric: metric_score})

        return loss

    def on_train_epoch_end(self):
        # log epoch metric

        loss = sum(output["loss"] for output in self.training_step_outputs) / len(
            self.training_step_outputs
        )
        self.logger.experiment.add_scalar("Loss/Train", loss, self.current_epoch)
        self.log("train_loss", loss, prog_bar=True)

        metric_score = sum(
            output[self.metric] for output in self.training_step_outputs
        ) / len(self.training_step_outputs)
        self.logger.experiment.add_scalar(
            f"{self.metric}/Train", metric_score, self.current_epoch
        )
        self.log(f"train_{self.metric}", metric_score, prog_bar=True)

        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        X, T = batch

        out = self.forward(X)
        loss = self.criterion(out, T)
        metric_score = self.metric_function(
            out,
            T,
        )

        self.validation_step_outputs.append({"loss": loss, self.metric: metric_score})

        return loss

    def on_validation_epoch_end(self):
        loss = sum(output["loss"] for output in self.validation_step_outputs) / len(
            self.validation_step_outputs
        )
        self.logger.experiment.add_scalar("Loss/Validation", loss, self.current_epoch)
        self.log("val_loss", loss, prog_bar=True)

        metric_score = sum(
            output[self.metric] for output in self.validation_step_outputs
        ) / len(self.validation_step_outputs)
        self.logger.experiment.add_scalar(
            f"{self.metric}/Validation", metric_score, self.current_epoch
        )
        self.log(f"val_{self.metric}", metric_score, prog_bar=True)

        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        return (
            torch.optim.SGD(self.parameters(), lr=self.lr)
            if self.opt == "sgd"
            else torch.optim.Adam(self.parameters(), lr=self.lr)
        )
