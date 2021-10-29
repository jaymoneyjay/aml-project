import torch
from tqdm import tqdm


class Trainer:
    def __init__(self, nb_epochs, run="run", verbose=True):
        """Create a trainer by specifying the number of epochs to train
        Args:
            nb_epochs: int. Number of epochs to train
            run: string. Title of the run to appear in tensorboard.
            verbose: bool. Whether or not to output training information.
        """
        self.nb_epochs = nb_epochs
        self.verbose = verbose

    #       self.tb = SummaryWriter(f'runs/{run}')

    def fit(self, model, dl_train, dl_val, verbose=True, lr=1e-3, optim='adam'):
        """Train the model on the specified data and print the training and validation loss and score.
        Args:
            model: Module. Model to train
            dl_train: DataLoader. DataLoader containing the training data
            dl_val: DataLoader. DataLoader containting the validation data
            verbose: bool. Whether or not to output training information
            lr: float. Learning rate
            optim: string. ID of the optimizer to use. ['adam', 'rmsprop']
        """

        self.verbose = verbose

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        model.to(device)

        optimizer = model.configure_optimizers(optim=optim, lr=lr)
        train_loss_epochs = []
        train_score_epochs = []
        val_score_epochs = []

        for e in tqdm(range(self.nb_epochs)):
            loss_train = []
            score_train = []
            for batch_idx, batch in enumerate(dl_train):
                model.train()
                optimizer.zero_grad()
                loss, score = model.training_step(batch, batch_idx, device)
                loss.backward()
                optimizer.step()

                loss_train.append(loss.item())
                score_train.append(score)

            loss_val = []
            score_val = []

            if self.verbose:
                for batch_idx, batch in enumerate(dl_val):
                    model.eval()
                    with torch.no_grad():
                        loss, score = model.validation_step(batch, batch_idx, device)
                        loss_val.append(loss.item())
                        score_val.append(score)
                avg_loss_train = round(sum(loss_train) / len(loss_train), 2)
                avg_score_train = round(sum(score_train) / len(score_train), 2)
                train_loss_epochs.append(avg_loss_train)
                train_score_epochs.append(avg_score_train)

                avg_loss_val = round(sum(loss_val) / len(loss_val), 2)
                avg_score_val = round(sum(score_val) / len(score_val), 2)
                val_score_epochs.append(avg_score_val)
                print(
                    f"# Epoch {e+1}/{self.nb_epochs}:\t loss={avg_loss_train}\t loss_val={avg_loss_val}\t score_val={avg_score_val}"
                )


        if self.verbose:
            return train_loss_epochs, train_score_epochs, val_score_epochs

    def test(self, model, dl_test, test_verbose=True, return_score=True):
        """Test the model on the specified data
        Args:
            model: Module. Model to train
            dl_test: DataLoader. DataLoader containting the test data
            test_verbose: bool. Whether the test result should be printed
        """

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        model.to(device)

        loss_test = []
        score_test = []
        for batch_idx, batch in enumerate(tqdm(dl_test)):
            model.eval()
            with torch.no_grad():
                loss, score = model.test_step(batch, batch_idx, device)
                loss_test.append(loss.item())
                score_test.append(score)

        avg_loss_test = round(sum(loss_test) / len(loss_test), 2)
        avg_score_test = round(sum(score_test) / len(score_test), 2)
        if test_verbose:
            print(f"loss_test={avg_loss_test}\t score_test={avg_score_test}")
        if return_score:
            return avg_score_test
