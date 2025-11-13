# Uni-modality Client Trainer
import copy
import operator
import clip
import torch.optim as optim
import torch.nn as nn
import torch.optim
import torch.utils.data
from imagebind.models import imagebind_model
from imagebind.data import load_and_transform_text, load_and_transform_vision_data, load_and_transform_audio_data
from model.ClientModel import BottomModel, TopModel
from imagebind.models.imagebind_model import ModalityType

class UMClientTrainer:
    def __init__(self, modality_type,  args):

        self.dataset = args.dataset
        self.modality_type = modality_type
        self.args = args
        self.bottom_model = BottomModel(n_f_in=args.n_f_in, n_f_out=args.n_f_out)

    # Set the model
    def set_model(self):
        # set the encoder
        if self.args.model == "imagebind":
            self.encoder = imagebind_model.imagebind_huge(pretrained=True)
        else:
            if self.modality_type == "img":
                if self.args.model == "clip":
                    self.encoder, _ = clip.load("ViT-B/32", device=self.args.device, jit=False)



    # Set the optimizer
    def set_optimizer(self):

        self.optimizer = optim.Adam(self.bottom_model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)


    def encode_data_imagebind(self, inputs):

        if self.modality_type == "img":
            images = load_and_transform_vision_data(inputs, self.args.device)
            return self.encoder({ModalityType.VISION: images})
        elif self.modality_type == "text":
            texts = load_and_transform_text(inputs, self.args.device)
            return self.encoder({ModalityType.TEXT: texts})
        elif self.modality_type == "audio":
            audios = load_and_transform_audio_data(inputs, self.args.device)
            return self.encoder({ModalityType.AUDIO: audios})
        else:
            raise ValueError("Unsupported modality type: {}".format(self.modality_type))


    # train the model
    def train(self, model, train_loader):
        model.train()
        total_loss = 0.0
        for images, texts in train_loader:
            images = images.to(self.args.device)
            texts = clip.tokenize(texts).to(self.args.device)

            self.optimizer.zero_grad()

            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)

            logits_per_image, logits_per_text = model(images, texts)
            loss = nn.CrossEntropyLoss()(logits_per_image, torch.arange(len(texts)).to(self.args.device))

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)



