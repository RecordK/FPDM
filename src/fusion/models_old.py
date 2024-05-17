import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
# from pytorch_metric_learning import losses
# from pytorch_metric_learning.distances import CosineSimilarity
from diffusers.models.controlnet import ControlNetConditioningEmbedding


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    0 : positivie
    1 : negative
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.cosine_similarity(output1, output2)
        # euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1 - label) * torch.pow(1 - euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - 1 - euclidean_distance, min=0.0),
                                                          2))
        return loss_contrastive


class Combiner(nn.Module):
    """
    reference : https://github.com/ABaldrati/CLIP4Cir/blob/master/src/combiner.py
    Combiner module which once trained fuses textual and visual information
    """

    def __init__(self, img_feature_dim: int, projection_dim: int, hidden_dim: int):
        """
        :param clip_feature_dim: CLIP input feature dimension
        :param projection_dim: projection dimension
        :param hidden_dim: hidden dimension
        """
        super(Combiner, self).__init__()
        self.kpt_projection_layer = nn.Linear(img_feature_dim, projection_dim)
        self.image_projection_layer = nn.Linear(img_feature_dim, projection_dim)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        self.combiner_layer = nn.Linear(projection_dim * 2, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, img_feature_dim)

        self.dropout3 = nn.Dropout(0.5)
        self.dynamic_scalar = nn.Sequential(nn.Linear(projection_dim * 2, hidden_dim),
                                            nn.ReLU(), nn.Dropout(0.5),
                                            nn.Linear(hidden_dim, 1), nn.Sigmoid())

        self.logit_scale = 100

    # def forward(self, image_features: torch.tensor, kpt_features: torch.tensor,
    #             target_features: torch.tensor) -> torch.tensor:
    def forward(self, image_features: torch.tensor, kpt_features: torch.tensor) -> torch.tensor:
        """
        Takes as input a triplet: image_features, text_features and target_features and outputs the logits which are
        the normalized dot product between the predicted features and the target_features.
        The logits are also multiplied by logit_scale parameter
        :param image_features: CLIP reference image features
        :param text_features: CLIP relative caption features
        :param target_features: CLIP target image features
        :return: scaled logits
        """
        predicted_features = F.normalize(self.combine_features(image_features, kpt_features), dim=-1)
        # target_features = F.normalize(target_features, dim=-1)
        # logits = self.logit_scale * predicted_features @ target_features.T
        # return logits
        return predicted_features

    def combine_features(self, image_features: torch.tensor, kpt_features: torch.tensor) -> torch.tensor:
        """
        Combine the reference image features and the caption features. It outputs the predicted features
        :param image_features: CLIP reference image features
        :param text_features: CLIP relative caption features
        :return: predicted features
        """
        kpt_projected_features = self.dropout1(F.relu(self.kpt_projection_layer(kpt_features)))
        image_projected_features = self.dropout2(F.relu(self.image_projection_layer(image_features)))

        raw_combined_features = torch.cat((kpt_projected_features, image_projected_features), -1)
        combined_features = self.dropout3(F.relu(self.combiner_layer(raw_combined_features)))
        dynamic_scalar = self.dynamic_scalar(raw_combined_features)
        output = self.output_layer(combined_features) + dynamic_scalar * kpt_features + (
                1 - dynamic_scalar) * image_features
        # return F.normalize(output, dim=-1)
        return output


class SimCLR(pl.LightningModule):
    def __init__(self, hidden_dim, lr, temperature, weight_decay, max_epochs=500):
        super().__init__()
        self.save_hyperparameters()
        assert self.hparams.temperature > 0.0, "The temperature must be a positive float!"
        # Base model f(.)
        self.convnet = torchvision.models.resnet18(
            pretrained=False, num_classes=4 * hidden_dim
        )  # num_classes is the output size of the last linear layer
        # The MLP for g(.) consists of Linear->ReLU->Linear
        self.convnet.fc = nn.Sequential(
            self.convnet.fc,  # Linear(ResNet output, 4*hidden_dim)
            nn.ReLU(inplace=True),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50
        )
        return [optimizer], [lr_scheduler]

    def info_nce_loss(self, batch, mode="train"):
        imgs = batch['img']
        imgs = torch.cat(imgs, dim=0)

        # Encode all images
        feats = self.convnet(imgs)
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.hparams.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        # Logging loss
        self.log(mode + "_loss", nll)
        # Get ranking position of positive example
        comb_sim = torch.cat(
            [cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -9e15)],  # First position positive example
            dim=-1,
        )
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Logging ranking metrics
        self.log(mode + "_acc_top1", (sim_argsort == 0).float().mean())
        self.log(mode + "_acc_top5", (sim_argsort < 5).float().mean())
        self.log(mode + "_acc_mean_pos", 1 + sim_argsort.float().mean())

        return nll

    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self.info_nce_loss(batch, mode="val")


class kptimgCLR(pl.LightningModule):
    def __init__(self, hidden_dim, lr, temperature, weight_decay, max_epochs=500):
        super().__init__()
        self.save_hyperparameters()
        assert self.hparams.temperature > 0.0, "The temperature must be a positive float!"
        # Base model f(.)
        self.img_convnet = torchvision.models.resnet18(
            pretrained=False, num_classes=4 * hidden_dim
        )  # num_classes is the output size of the last linear layer
        # The MLP for g(.) consists of Linear->ReLU->Linear
        self.img_convnet.fc = nn.Sequential(
            self.img_convnet.fc,  # Linear(ResNet output, 4*hidden_dim)
            nn.ReLU(inplace=True),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

        self.kpt_convnet = torchvision.models.resnet18(
            pretrained=False, num_classes=4 * hidden_dim
        )  # num_classes is the output size of the last linear layer
        # The MLP for g(.) consists of Linear->ReLU->Linear
        # self.kpt_convnet.conv1 = nn.Conv2d(20, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.kpt_convnet.fc = nn.Sequential(
            self.kpt_convnet.fc,  # Linear(ResNet output, 4*hidden_dim)
            nn.ReLU(inplace=True),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

        # self.loss_triplet = nn.TripletMarginWithDistanceLoss(distance_function = lambda x, y: 1.0 - F.cosine_similarity(x, y))
        self.loss_triplet = nn.TripletMarginLoss(margin=0.5)
        self.loss_cos = nn.MSELoss()
        # self.loss_cos = ContrastiveLoss(margin=0.5)
        # self.loss_cos = nn.CosineSimilarity()

    def freeze_model(self, model, freeze=True):
        if freeze:
            for param in model.parameters():
                param.requires_grad = False
        else:
            for param in model.parameters():
                param.requires_grad = True

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50
        )
        return [optimizer], [lr_scheduler]

    def contrastive_loss(self, mode="train"):
        _loss = self.loss_cos(self.anchor_features, self.kpt_features)  # .detach()
        # _loss = self.loss_cos(self.anchor_features, self.kpt_features, self.kpt_label) # .detach()
        self.log(mode + "-cos-loss", _loss.float().mean())
        return _loss.float().mean()

    def triplet_loss(self, mode="train"):
        _loss = self.loss_triplet(self.anchor_features,
                                  self.pos_features,
                                  self.neg_features)
        self.log(mode + "-triplet-loss", _loss.float().mean())
        return _loss.float().mean()

    def training_step(self, batch, batch_idx):
        anchor_img = batch['anchor_img']
        pos_img = batch['pos_img']
        neg_img = batch['neg_img']
        kpt_img = batch['kpt_img']
        # self.kpt_label = batch['kpt_label']

        self.anchor_features = self.img_convnet(anchor_img)
        self.pos_features = self.img_convnet(pos_img)
        self.neg_features = self.img_convnet(neg_img)
        self.kpt_features = self.kpt_convnet(kpt_img)

        loss_cos = self.contrastive_loss('train')
        loss_triplet = self.triplet_loss('train')
        _total_loss = loss_triplet + loss_cos
        self.log('train' + "-total-loss", _total_loss.float().mean())
        return _total_loss

    def validation_step(self, batch, batch_idx):  # kpt 정닶 없음.. 아직
        anchor_img = batch['anchor_img']
        pos_img = batch['pos_img']
        neg_img = batch['neg_img']
        kpt_img = batch['kpt_img']
        # self.kpt_label = batch['kpt_label']

        self.anchor_features = self.img_convnet(anchor_img)
        self.pos_features = self.img_convnet(pos_img)
        self.neg_features = self.img_convnet(neg_img)
        self.kpt_features = self.kpt_convnet(kpt_img)

        loss_cos = self.contrastive_loss('val')
        loss_triplet = self.triplet_loss('val')
        _total_loss = loss_triplet + loss_cos
        self.log('val' + "total-loss", _total_loss.float().mean())
        return _total_loss


class KptImgSimCLR(pl.LightningModule):
    def __init__(self, hidden_dim, lr, temperature, weight_decay, max_epochs=500):
        super().__init__()
        self.save_hyperparameters()
        assert self.hparams.temperature > 0.0, "The temperature must be a positive float!"
        # Base model f(.)
        self.img_convnet = torchvision.models.resnet18(
            pretrained=False, num_classes=4 * hidden_dim
        )  # num_classes is the output size of the last linear layer
        # The MLP for g(.) consists of Linear->ReLU->Linear
        self.img_convnet.fc = nn.Sequential(
            self.img_convnet.fc,  # Linear(ResNet output, 4*hidden_dim)
            nn.ReLU(inplace=True),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

        self.kpt_convnet = torchvision.models.resnet18(
            pretrained=False, num_classes=4 * hidden_dim
        )  # num_classes is the output size of the last linear layer
        # The MLP for g(.) consists of Linear->ReLU->Linear
        # self.kpt_convnet.conv1 = nn.Conv2d(20, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.kpt_convnet.fc = nn.Sequential(
            self.kpt_convnet.fc,  # Linear(ResNet output, 4*hidden_dim)
            nn.ReLU(inplace=True),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50
        )
        return [optimizer], [lr_scheduler]

    def imgs_info_nce_loss(self, img_feats, mode="train"):
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(img_feats[:, None, :], img_feats[None, :, :], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.hparams.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        # Logging loss
        self.log(mode + "_loss", nll)
        # Get ranking position of positive example
        comb_sim = torch.cat(
            [cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -9e15)],  # First position positive example
            dim=-1,
        )
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Logging ranking metrics
        self.log(mode + "_img_acc_top1", (sim_argsort == 0).float().mean())
        self.log(mode + "_img_acc_top5", (sim_argsort < 5).float().mean())
        self.log(mode + "_img_acc_mean_pos", 1 + sim_argsort.float().mean())
        return nll

    def kpts_info_nce_loss(self, img_feats, kpt_feats, mode="train"):
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(img_feats[:, None, :], kpt_feats[None, :, :], dim=-1)
        # Mask out cosine similarity to itself
        pos_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        # cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        # pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.hparams.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        # Logging loss
        self.log(mode + "_loss", nll)
        # Get ranking position of positive example
        comb_sim = torch.cat(
            [cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -9e15)],  # First position positive example
            dim=-1,
        )
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Logging ranking metrics
        self.log(mode + "_kpt_acc_top1", (sim_argsort == 0).float().mean())
        self.log(mode + "_kpt_acc_top5", (sim_argsort < 5).float().mean())
        self.log(mode + "_kpt_acc_mean_pos", 1 + sim_argsort.float().mean())
        return nll

    def training_step(self, batch, batch_idx):
        anchor_imgs_list = batch['anchor_img']
        pos_imgs_list = batch['pos_img']
        kpts = batch['kpt']
        batch_size = kpts.shape[0]

        imgs = torch.cat((anchor_imgs_list, pos_imgs_list), dim=0)
        img_feats = self.img_convnet(imgs)
        kpt_feats = self.kpt_convnet(kpts)

        img_feats = F.normalize(img_feats)
        kpt_feats = F.normalize(kpt_feats)

        img_info_ncs_loss = self.imgs_info_nce_loss(img_feats, mode="train")
        kpt_info_ncs_loss = self.kpts_info_nce_loss(img_feats[:batch_size], kpt_feats, mode="train")
        total_loss = img_info_ncs_loss + kpt_info_ncs_loss
        self.log('train' + "_total_loss", 1 + total_loss.float().mean())
        return total_loss

    def validation_step(self, batch, batch_idx):
        anchor_imgs_list = batch['anchor_img']
        pos_imgs_list = batch['pos_img']
        kpts = batch['kpt']

        imgs = torch.cat((anchor_imgs_list, pos_imgs_list), dim=0)
        batch_size = kpts.shape[0]

        img_feats = self.img_convnet(imgs)
        kpt_feats = self.kpt_convnet(kpts)

        img_feats = F.normalize(img_feats)
        kpt_feats = F.normalize(kpt_feats)

        img_info_ncs_loss = self.imgs_info_nce_loss(img_feats, mode="val")
        kpt_info_ncs_loss = self.kpts_info_nce_loss(img_feats[:batch_size], kpt_feats, mode="val")
        total_loss = img_info_ncs_loss + kpt_info_ncs_loss
        self.log('val' + "_total_loss", 1 + total_loss.float().mean())
        return total_loss


class FusionSimCLR(pl.LightningModule):
    def __init__(self, hidden_dim, lr, temperature, weight_decay, max_epochs=500):
        super().__init__()
        self.save_hyperparameters()
        assert self.hparams.temperature > 0.0, "The temperature must be a positive float!"
        # Base model f(.)
        # pretrained_model = KptImgSimCLR.load_from_checkpoint(
        #     './logs/kptimg-infonce-learning/2024-04-29T01-18-53/last.ckpt') ## ?????????????  , simcrl 2024-04-29T01-17-45 # triplet 2024-04-29T01-18-53
        # img_encoder = pretrained_model.img_convnet
        # kpt_encoder = pretrained_model.kpt_convnet
        # self.img_encoder = img_encoder
        # self.kpt_encoder = kpt_encoder

        self.img_encoder = torchvision.models.resnet18(
            pretrained=False, num_classes=4 * hidden_dim
        )  # num_classes is the output size of the last linear layer
        # The MLP for g(.) consists of Linear->ReLU->Linear
        self.img_encoder.fc = nn.Sequential(
            self.img_encoder.fc,  # Linear(ResNet output, 4*hidden_dim)
            nn.ReLU(inplace=True),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

        self.kpt_encoder = torchvision.models.resnet18(
            pretrained=False, num_classes=4 * hidden_dim
        )  # num_classes is the output size of the last linear layer
        # The MLP for g(.) consists of Linear->ReLU->Linear
        # self.kpt_convnet.conv1 = nn.Conv2d(20, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.kpt_encoder.fc = nn.Sequential(
            self.kpt_encoder.fc,  # Linear(ResNet output, 4*hidden_dim)
            nn.ReLU(inplace=True),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.combiner = Combiner(img_feature_dim=512, projection_dim=512, hidden_dim=512)

        # self.freeze_model(img_encoder, freeze=False)
        # self.freeze_model(kpt_encoder, freeze=False)

    def freeze_model(self, model, freeze=True):
        if freeze:
            for param in model.parameters():
                param.requires_grad = False
        else:
            for param in model.parameters():
                param.requires_grad = True

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50
        )
        return [optimizer], [lr_scheduler]

    def fusion_info_nce_loss(self, img_feats, fushion_feats, mode="train"):
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(img_feats[:, None, :], fushion_feats[None, :, :], dim=-1)
        # Mask out cosine similarity to itself
        pos_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        # cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        # pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.hparams.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)

        # nll =  torch.mean(torch.abs(img_feats - fushion_feats), dim=-1)
        nll = nll.mean()

        # Logging loss
        self.log(mode + "_loss", nll)
        # Get ranking position of positive example
        comb_sim = torch.cat(
            [cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -9e15)],  # First position positive example
            dim=-1,
        )
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Logging ranking metrics
        self.log(mode + "_fushion_acc_top1", (sim_argsort == 0).float().mean())
        self.log(mode + "_fushion_acc_top5", (sim_argsort < 5).float().mean())
        self.log(mode + "_fushion_acc_mean_pos", 1 + sim_argsort.float().mean())
        return nll

    def training_step(self, batch, batch_idx):
        reference_imgs = batch['reference_img']
        target_imgs = batch['target_img']
        target_kpts = batch['target_kpt']

        refer_img_feats = self.img_encoder(reference_imgs)
        targets_img_feats = self.img_encoder(target_imgs)
        target_kpts_feats = self.kpt_encoder(target_kpts)

        fusion_img_feats = self.combiner(refer_img_feats, target_kpts_feats)
        info_ncs_loss = self.fusion_info_nce_loss(targets_img_feats, fusion_img_feats, mode="train")
        total_loss = info_ncs_loss
        self.log('train' + "_total_loss", total_loss.float().mean())
        return total_loss

    def validation_step(self, batch, batch_idx):
        reference_imgs = batch['reference_img']
        target_imgs = batch['target_img']
        target_kpts = batch['target_kpt']

        refer_img_feats = self.img_encoder(reference_imgs)
        targets_img_feats = self.img_encoder(target_imgs)
        target_kpts_feats = self.kpt_encoder(target_kpts)
        fusion_img_feats = self.combiner(refer_img_feats, target_kpts_feats)
        info_ncs_loss = self.fusion_info_nce_loss(targets_img_feats, fusion_img_feats, mode="val")
        total_loss = info_ncs_loss
        self.log('val' + "_total_loss", total_loss.float().mean())
        return total_loss


class PoseProjModel(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class FusionSimCLR_AD(pl.LightningModule):
    def __init__(self, hidden_dim, lr, temperature, weight_decay, max_epochs=500):
        super().__init__()
        self.save_hyperparameters()
        assert self.hparams.temperature > 0.0, "The temperature must be a positive float!"

        self.pose_encoder = ControlNetConditioningEmbedding(
            conditioning_embedding_channels=320,
            block_out_channels=(16, 32, 96, 256),
            conditioning_channels=3)

        self.img_encoder = torchvision.models.resnet18(
            pretrained=True)  # num_classes is the output size of the last linear layer
        # self.img_encoder.fc = nn.Identity()
        self.img_encoder.avgpool = nn.Flatten()
        self.img_encoder.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(32768, hidden_dim),
        )

        self.kpt_encoder = torchvision.models.resnet18(
            pretrained=False)  # num_classes is the output size of the last linear layer
        self.kpt_encoder.avgpool = nn.Flatten()
        self.kpt_encoder.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(32768, hidden_dim),
        )
        self.combiner = Combiner(img_feature_dim=hidden_dim, projection_dim=768, hidden_dim=768)  # 768

        # self.freeze_model(img_encoder, freeze=False)
        # self.freeze_model(kpt_encoder, freeze=False)

    def freeze_model(self, model, freeze=True):
        if freeze:
            for param in model.parameters():
                param.requires_grad = False
        else:
            for param in model.parameters():
                param.requires_grad = True

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50
        )
        return [optimizer], [lr_scheduler]

    def fusion_info_nce_loss(self, img_feats, fushion_feats, mode="train"):
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(img_feats[:, None, :], fushion_feats[None, :, :], dim=-1)
        # Mask out cosine similarity to itself
        pos_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        # cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        # pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.hparams.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)

        # nll =  torch.mean(torch.abs(img_feats - fushion_feats), dim=-1)
        nll = nll.mean()

        # Logging loss
        self.log(mode + "_loss", nll)
        # Get ranking position of positive example
        comb_sim = torch.cat(
            [cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -9e15)],  # First position positive example
            dim=-1,
        )
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Logging ranking metrics
        self.log(mode + "_fushion_acc_top1", (sim_argsort == 0).float().mean())
        self.log(mode + "_fushion_acc_top5", (sim_argsort < 5).float().mean())
        self.log(mode + "_fushion_acc_mean_pos", 1 + sim_argsort.float().mean())
        return nll

    def training_step(self, batch, batch_idx):
        reference_imgs = batch['reference_img']
        target_imgs = batch['target_img']
        target_kpts = batch['target_pose']

        refer_img_feats = self.img_encoder(reference_imgs)
        targets_img_feats = self.img_encoder(target_imgs)
        target_kpts_feats = self.kpt_encoder(target_kpts)

        fusion_img_feats = self.combiner(refer_img_feats, target_kpts_feats)
        info_ncs_loss = self.fusion_info_nce_loss(targets_img_feats, fusion_img_feats, mode="train")
        total_loss = info_ncs_loss
        self.log('train' + "_total_loss", total_loss.float().mean())
        return total_loss

    def validation_step(self, batch, batch_idx):
        reference_imgs = batch['reference_img']
        target_imgs = batch['target_img']
        target_kpts = batch['target_pose']

        self.pose_encoder
        refer_img_feats = self.img_encoder(reference_imgs)
        targets_img_feats = self.img_encoder(target_imgs)
        target_kpts_feats = self.kpt_encoder(target_kpts)
        fusion_img_feats = self.combiner(refer_img_feats, target_kpts_feats)
        info_ncs_loss = self.fusion_info_nce_loss(targets_img_feats, fusion_img_feats, mode="val")
        total_loss = info_ncs_loss
        self.log('val' + "_total_loss", total_loss.float().mean())
        return total_loss


class FusionSimCLR_pool(pl.LightningModule):
    def __init__(self, hidden_dim, lr, temperature, weight_decay, max_epochs=500):
        super().__init__()
        self.save_hyperparameters()
        assert self.hparams.temperature > 0.0, "The temperature must be a positive float!"
        # Base model f(.)
        # pretrained_model = KptImgSimCLR.load_from_checkpoint(
        #     './logs/kptimg-infonce-learning/2024-04-29T01-18-53/last.ckpt') ## ?????????????  , simcrl 2024-04-29T01-17-45 # triplet 2024-04-29T01-18-53
        # img_encoder = pretrained_model.img_convnet
        # kpt_encoder = pretrained_model.kpt_convnet
        # self.img_encoder = img_encoder
        # self.kpt_encoder = kpt_encoder

        self.img_encoder = torchvision.models.resnet18(
            pretrained=False)  # num_classes is the output size of the last linear layer
        # self.img_encoder.fc = nn.Identity()
        self.img_encoder.avgpool = nn.Flatten()
        self.img_encoder.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(32768, hidden_dim),
        )

        self.kpt_encoder = torchvision.models.resnet18(
            pretrained=False)  # num_classes is the output size of the last linear layer
        self.kpt_encoder.avgpool = nn.Flatten()
        self.kpt_encoder.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(32768, hidden_dim),
        )
        self.combiner = Combiner(img_feature_dim=hidden_dim, projection_dim=512, hidden_dim=512)  # 768

        # self.freeze_model(img_encoder, freeze=False)
        # self.freeze_model(kpt_encoder, freeze=False)

    def freeze_model(self, model, freeze=True):
        if freeze:
            for param in model.parameters():
                param.requires_grad = False
        else:
            for param in model.parameters():
                param.requires_grad = True

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50
        )
        return [optimizer], [lr_scheduler]

    def fusion_info_nce_loss(self, img_feats, fushion_feats, mode="train"):
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(img_feats[:, None, :], fushion_feats[None, :, :], dim=-1)
        # Mask out cosine similarity to itself
        pos_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        # cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        # pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.hparams.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)

        # nll =  torch.mean(torch.abs(img_feats - fushion_feats), dim=-1)
        nll = nll.mean()

        # Logging loss
        self.log(mode + "_loss", nll)
        # Get ranking position of positive example
        comb_sim = torch.cat(
            [cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -9e15)],  # First position positive example
            dim=-1,
        )
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Logging ranking metrics
        self.log(mode + "_fushion_acc_top1", (sim_argsort == 0).float().mean())
        self.log(mode + "_fushion_acc_top5", (sim_argsort < 5).float().mean())
        self.log(mode + "_fushion_acc_mean_pos", 1 + sim_argsort.float().mean())
        return nll

    def training_step(self, batch, batch_idx):
        reference_imgs = batch['reference_img']
        target_imgs = batch['target_img']
        target_kpts = batch['target_kpt']

        refer_img_feats = self.img_encoder(reference_imgs)
        targets_img_feats = self.img_encoder(target_imgs)
        target_kpts_feats = self.kpt_encoder(target_kpts)

        fusion_img_feats = self.combiner(refer_img_feats, target_kpts_feats)
        info_ncs_loss = self.fusion_info_nce_loss(targets_img_feats, fusion_img_feats, mode="train")
        total_loss = info_ncs_loss
        self.log('train' + "_total_loss", total_loss.float().mean())
        return total_loss

    def validation_step(self, batch, batch_idx):
        reference_imgs = batch['reference_img']
        target_imgs = batch['target_img']
        target_kpts = batch['target_kpt']

        refer_img_feats = self.img_encoder(reference_imgs)
        targets_img_feats = self.img_encoder(target_imgs)
        target_kpts_feats = self.kpt_encoder(target_kpts)
        fusion_img_feats = self.combiner(refer_img_feats, target_kpts_feats)
        info_ncs_loss = self.fusion_info_nce_loss(targets_img_feats, fusion_img_feats, mode="val")
        total_loss = info_ncs_loss
        self.log('val' + "_total_loss", total_loss.float().mean())
        return total_loss
