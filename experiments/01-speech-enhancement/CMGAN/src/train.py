from models.generator import TSCNet
from models import discriminator
import os
from data import dataloader
import torch.nn.functional as F
import torch
from utils import power_compress, power_uncompress
import logging
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=120, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--log_interval", type=int, default=500)
parser.add_argument("--decay_epoch", type=int, default=30, help="epoch from which to start lr decay")
parser.add_argument("--init_lr", type=float, default=5e-4, help="initial learning rate")
parser.add_argument("--cut_len", type=int, default=16000*2, help="cut length, default is 2 seconds in denoise "
                                                                 "and dereverberation")
parser.add_argument("--data_dir", type=str, default='dir to VCTK-DEMAND dataset',
                    help="dir of VCTK+DEMAND dataset")
parser.add_argument("--save_model_dir", type=str, default='./saved_model',
                    help="dir of saved model")
parser.add_argument("--tensorboard_dir", type=str, default='./runs',
                    help="dir for TensorBoard event files")
parser.add_argument("--loss_weights", type=list, default=[0.1, 0.9, 0.2, 0.05],
                    help="weights of RI components, magnitude, time loss, and Metric Disc")
args = parser.parse_args()
logging.basicConfig(level=logging.INFO)

if args.log_interval < 1:
    raise ValueError("--log_interval must be at least 1")


class Trainer:
    def __init__(self, train_ds, test_ds, device, writer):
        self.n_fft = 400
        self.hop = 100
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.device = device
        self.writer = writer
        self.model = TSCNet(
            num_channel=64, num_features=self.n_fft // 2 + 1
        ).to(self.device)
        summary(
            self.model, [(1, 2, args.cut_len // self.hop + 1, int(self.n_fft / 2) + 1)]
        )
        self.discriminator = discriminator.Discriminator(ndf=16).to(self.device)
        summary(
            self.discriminator,
            [
                (1, 1, int(self.n_fft / 2) + 1, args.cut_len // self.hop + 1),
                (1, 1, int(self.n_fft / 2) + 1, args.cut_len // self.hop + 1),
            ],
        )
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.init_lr)
        self.optimizer_disc = torch.optim.AdamW(
            self.discriminator.parameters(), lr=2 * args.init_lr
        )

    def forward_generator_step(self, clean, noisy):

        # Normalization
        c = torch.sqrt(noisy.size(-1) / torch.sum((noisy**2.0), dim=-1))
        noisy, clean = torch.transpose(noisy, 0, 1), torch.transpose(clean, 0, 1)
        noisy, clean = torch.transpose(noisy * c, 0, 1), torch.transpose(
            clean * c, 0, 1
        )

        noisy_stft = torch.stft(
            noisy,
            self.n_fft,
            self.hop,
            window=torch.hamming_window(self.n_fft).to(self.device),
            onesided=True,
            return_complex=True,
        )
        clean_stft = torch.stft(
            clean,
            self.n_fft,
            self.hop,
            window=torch.hamming_window(self.n_fft).to(self.device),
            onesided=True,
            return_complex=True,
        )
        noisy_spec = torch.view_as_real(noisy_stft)
        clean_spec = torch.view_as_real(clean_stft)
        noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
        clean_spec = power_compress(clean_spec)
        clean_real = clean_spec[:, 0, :, :].unsqueeze(1)
        clean_imag = clean_spec[:, 1, :, :].unsqueeze(1)

        est_real, est_imag = self.model(noisy_spec)
        est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)
        est_mag = torch.sqrt(est_real**2 + est_imag**2)
        clean_mag = torch.sqrt(clean_real**2 + clean_imag**2)

        est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
        est_audio = torch.istft(
            torch.view_as_complex(est_spec_uncompress.contiguous()),
            self.n_fft,
            self.hop,
            window=torch.hamming_window(self.n_fft).to(self.device),
            onesided=True,
        )

        return {
            "est_real": est_real,
            "est_imag": est_imag,
            "est_mag": est_mag,
            "clean_real": clean_real,
            "clean_imag": clean_imag,
            "clean_mag": clean_mag,
            "est_audio": est_audio,
        }

    def calculate_generator_loss(self, generator_outputs):

        predict_fake_metric = self.discriminator(
            generator_outputs["clean_mag"], generator_outputs["est_mag"]
        )
        gen_loss_GAN = F.mse_loss(
            predict_fake_metric.flatten(), generator_outputs["one_labels"].float()
        )

        loss_mag = F.mse_loss(
            generator_outputs["est_mag"], generator_outputs["clean_mag"]
        )
        loss_ri = F.mse_loss(
            generator_outputs["est_real"], generator_outputs["clean_real"]
        ) + F.mse_loss(generator_outputs["est_imag"], generator_outputs["clean_imag"])

        time_loss = torch.mean(
            torch.abs(generator_outputs["est_audio"] - generator_outputs["clean"])
        )

        weighted_losses = {
            "ri_weighted": args.loss_weights[0] * loss_ri,
            "magnitude_weighted": args.loss_weights[1] * loss_mag,
            "time_weighted": args.loss_weights[2] * time_loss,
            "adversarial_weighted": args.loss_weights[3] * gen_loss_GAN,
        }
        loss = sum(weighted_losses.values())
        loss_details = {
            "ri_raw": loss_ri,
            "magnitude_raw": loss_mag,
            "time_raw": time_loss,
            "adversarial_raw": gen_loss_GAN,
            **weighted_losses,
        }

        return loss, loss_details

    @staticmethod
    def metrics_to_scalars(generator_loss, loss_details, discriminator_loss):
        metrics = {
            "generator_total": generator_loss,
            **loss_details,
            "discriminator": discriminator_loss,
        }
        names = list(metrics)
        values = torch.stack(
            [value.detach().reshape(()) for value in metrics.values()]
        ).cpu().tolist()
        return dict(zip(names, values))

    def calculate_discriminator_loss(self, generator_outputs):

        length = generator_outputs["est_audio"].size(-1)
        est_audio_list = list(generator_outputs["est_audio"].detach().cpu().numpy())
        clean_audio_list = list(generator_outputs["clean"].cpu().numpy()[:, :length])
        pesq_score = discriminator.batch_pesq(clean_audio_list, est_audio_list)

        # The calculation of PESQ can be None due to silent part
        if pesq_score is not None:
            predict_enhance_metric = self.discriminator(
                generator_outputs["clean_mag"], generator_outputs["est_mag"].detach()
            )
            predict_max_metric = self.discriminator(
                generator_outputs["clean_mag"], generator_outputs["clean_mag"]
            )
            discrim_loss_metric = F.mse_loss(
                predict_max_metric.flatten(), generator_outputs["one_labels"]
            ) + F.mse_loss(predict_enhance_metric.flatten(), pesq_score)
        else:
            discrim_loss_metric = None

        return discrim_loss_metric

    def train_step(self, batch):

        # Trainer generator
        clean = batch[0].to(self.device)
        noisy = batch[1].to(self.device)
        one_labels = torch.ones(clean.size(0), device=self.device)

        generator_outputs = self.forward_generator_step(
            clean,
            noisy,
        )
        generator_outputs["one_labels"] = one_labels
        generator_outputs["clean"] = clean

        loss, loss_details = self.calculate_generator_loss(generator_outputs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Train Discriminator
        discrim_loss_metric = self.calculate_discriminator_loss(generator_outputs)

        if discrim_loss_metric is not None:
            self.optimizer_disc.zero_grad()
            discrim_loss_metric.backward()
            self.optimizer_disc.step()
        else:
            discrim_loss_metric = torch.zeros((), device=self.device)

        return self.metrics_to_scalars(loss, loss_details, discrim_loss_metric)

    @torch.no_grad()
    def test_step(self, batch):

        clean = batch[0].to(self.device)
        noisy = batch[1].to(self.device)
        one_labels = torch.ones(clean.size(0), device=self.device)

        generator_outputs = self.forward_generator_step(
            clean,
            noisy,
        )
        generator_outputs["one_labels"] = one_labels
        generator_outputs["clean"] = clean

        loss, loss_details = self.calculate_generator_loss(generator_outputs)

        discrim_loss_metric = self.calculate_discriminator_loss(generator_outputs)
        if discrim_loss_metric is None:
            discrim_loss_metric = torch.zeros((), device=self.device)

        return self.metrics_to_scalars(loss, loss_details, discrim_loss_metric)

    def test(self):
        self.model.eval()
        self.discriminator.eval()
        totals = {}
        num_batches = 0
        for batch in self.test_ds:
            metrics = self.test_step(batch)
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + value
            num_batches += 1

        if num_batches == 0:
            raise ValueError("The test dataset contains no batches.")
        averages = {name: value / num_batches for name, value in totals.items()}

        template = "Device: {}, Generator loss: {}, Discriminator loss: {}"
        logging.info(
            template.format(
                self.device,
                averages["generator_total"],
                averages["discriminator"],
            )
        )

        return averages

    def train(self):
        scheduler_G = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=args.decay_epoch, gamma=0.5
        )
        scheduler_D = torch.optim.lr_scheduler.StepLR(
            self.optimizer_disc, step_size=args.decay_epoch, gamma=0.5
        )
        num_train_batches = len(self.train_ds)
        if num_train_batches == 0:
            raise ValueError("The training dataset contains no complete batches.")

        for epoch in range(args.epochs):
            self.model.train()
            self.discriminator.train()
            train_totals = {}
            for idx, batch in enumerate(self.train_ds):
                step = idx + 1
                metrics = self.train_step(batch)
                for name, value in metrics.items():
                    train_totals[name] = train_totals.get(name, 0.0) + value

                if (step % args.log_interval) == 0:
                    template = "Device: {}, Epoch {}, Step {}, loss: {}, disc_loss: {}"
                    logging.info(
                        template.format(
                            self.device,
                            epoch,
                            step,
                            metrics["generator_total"],
                            metrics["discriminator"],
                        )
                    )
                    global_step = epoch * num_train_batches + step
                    for name, value in metrics.items():
                        self.writer.add_scalar(f"train/step/{name}", value, global_step)

            train_averages = {
                name: value / num_train_batches
                for name, value in train_totals.items()
            }
            for name, value in train_averages.items():
                self.writer.add_scalar(f"train/epoch/{name}", value, epoch + 1)

            test_metrics = self.test()
            for name, value in test_metrics.items():
                self.writer.add_scalar(f"test/epoch/{name}", value, epoch + 1)
            self.writer.add_scalar(
                "learning_rate/generator",
                self.optimizer.param_groups[0]["lr"],
                epoch + 1,
            )
            self.writer.add_scalar(
                "learning_rate/discriminator",
                self.optimizer_disc.param_groups[0]["lr"],
                epoch + 1,
            )
            self.writer.flush()

            gen_loss = test_metrics["generator_total"]
            path = os.path.join(
                args.save_model_dir,
                "CMGAN_epoch_" + str(epoch) + "_" + str(gen_loss)[:5],
            )
            if not os.path.exists(args.save_model_dir):
                os.makedirs(args.save_model_dir)
            torch.save(self.model.state_dict(), path)
            scheduler_G.step()
            scheduler_D.step()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CMGAN training requires a CUDA-capable GPU.")

    device = torch.device("cuda:0")
    print(args)
    print("Using GPU:", torch.cuda.get_device_name(device))
    train_ds, test_ds = dataloader.load_data(
        args.data_dir, args.batch_size, 2, args.cut_len
    )
    writer = SummaryWriter(log_dir=args.tensorboard_dir)
    try:
        trainer = Trainer(train_ds, test_ds, device, writer)
        trainer.train()
    finally:
        writer.close()


if __name__ == "__main__":
    main()
