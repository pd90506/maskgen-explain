import torch
from torch import nn
from transformers import ViTModel, ViTConfig, ViTPreTrainedModel
from transformers.modeling_outputs import ModelOutput, dataclass
import math
from typing import Optional, Tuple
import torch.nn.functional as F

@dataclass
class ImageMultitaskOutput(ModelOutput):
    """
    Base class for outputs of image classification models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each stage) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states
            (also called feature maps) of the model at the output of each stage.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    ce_loss: Optional[torch.FloatTensor] = None
    mim_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    reconstructed_pixel_values: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
# Note: Config needs to be defined or loaded with the appropriate attributes.

    

class ViTForMultitask(ViTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.vit = ViTModel(config, add_pooling_layer=False, use_mask_token=True)
        
        # Classification head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # MIM head
        self.mim_decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=config.hidden_size,
                out_channels=config.encoder_stride**2 * config.num_channels,
                kernel_size=1,
            ),
            nn.PixelShuffle(config.encoder_stride)
        )

        self.init_weights()

    def forward(
        self,
        pixel_values,
        labels=None,
        bool_masked_pos=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        interpolate_pos_encoding=None,
        return_dict=None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if bool_masked_pos is not None and (self.config.patch_size != self.config.encoder_stride):
            raise ValueError(
                "When `bool_masked_pos` is provided, `patch_size` must be equal to `encoder_stride` to ensure that "
                "the reconstructed image has the same dimensions as the input. "
                f"Got `patch_size` = {self.config.patch_size} and `encoder_stride` = {self.config.encoder_stride}."
            )

        # Obtain vit outputs
        outputs = self.vit(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )
        if outputs is None or outputs[0] is None:
            raise ValueError("The ViT model did not return expected outputs.")

        sequence_output = outputs[0]
        sequence_output = sequence_output / sequence_output.norm(p=2, dim=-1, keepdim=True)

        logits = self.classifier(sequence_output[:, 0, :])

        loss = None
        ce_loss = None
        mim_loss = None
        reconstructed_pixel_values = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            ce_loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            loss = ce_loss

            # Calculate MIM loss only when label is available
            # Reshape to (batch_size, num_channels, height, width)
            sequence_output = sequence_output[:, 1:]
            batch_size, sequence_length, num_channels = sequence_output.shape
            height = width = math.floor(sequence_length**0.5)
            sequence_output = sequence_output.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)
    
            # Reconstruct pixel values
            reconstructed_pixel_values = self.mim_decoder(sequence_output)
            
            if bool_masked_pos is not None:
                size = self.config.image_size // self.config.patch_size
                bool_masked_pos = bool_masked_pos.reshape(-1, size, size)
                mask = (
                    bool_masked_pos.repeat_interleave(self.config.patch_size, 1)
                    .repeat_interleave(self.config.patch_size, 2)
                    .unsqueeze(1)
                    .contiguous()
                )
                reconstruction_loss = nn.functional.l1_loss(pixel_values, reconstructed_pixel_values, reduction="none")
                mim_loss = (reconstruction_loss * mask).sum() / (mask.sum() + 1e-5) / self.config.num_channels
                loss += mim_loss


        if not return_dict:
            output = (logits, reconstructed_pixel_values) + outputs[2:]
            return ((loss, mim_loss) + output) if loss is not None else output

        return ImageMultitaskOutput(
            loss=loss,
            logits=logits,
            # ce_loss=ce_loss,
            # mim_loss=mim_loss,
            # reconstructed_pixel_values=reconstructed_pixel_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# Note: Config needs to be defined or loaded with the appropriate attributes.



class ViTForImageClassification2(ViTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.vit = ViTModel(config, add_pooling_layer=False, use_mask_token=False)
        
        # Classification head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        self.init_weights()

    def forward(
        self,
        pixel_values,
        labels=None,
        bool_masked_pos=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        interpolate_pos_encoding=None,
        return_dict=None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if bool_masked_pos is not None and (self.config.patch_size != self.config.encoder_stride):
            raise ValueError(
                "When `bool_masked_pos` is provided, `patch_size` must be equal to `encoder_stride` to ensure that "
                "the reconstructed image has the same dimensions as the input. "
                f"Got `patch_size` = {self.config.patch_size} and `encoder_stride` = {self.config.encoder_stride}."
            )

        # Obtain vit outputs
        outputs = self.vit(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )
        if outputs is None or outputs[0] is None:
            raise ValueError("The ViT model did not return expected outputs.")

        sequence_output = outputs[0]
        sequence_output = sequence_output / sequence_output.norm(p=2, dim=-1, keepdim=True)

        logits = self.classifier(sequence_output[:, 0, :])

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ImageMultitaskOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# Note: Config needs to be defined or loaded with the appropriate attributes.



class ViTForMultitask3(ViTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.vit = ViTModel(config, add_pooling_layer=False, use_mask_token=False)
        
        # Classification head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # MIM head
        self.mim_decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=config.hidden_size,
                out_channels=config.encoder_stride**2 * config.num_channels,
                kernel_size=1,
            ),
            nn.PixelShuffle(config.encoder_stride)
        )

        self.init_weights()

    def forward(
        self,
        pixel_values,
        labels=None,
        bool_masked_pos=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        interpolate_pos_encoding=None,
        return_dict=None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if bool_masked_pos is not None and (self.config.patch_size != self.config.encoder_stride):
            raise ValueError(
                "When `bool_masked_pos` is provided, `patch_size` must be equal to `encoder_stride` to ensure that "
                "the reconstructed image has the same dimensions as the input. "
                f"Got `patch_size` = {self.config.patch_size} and `encoder_stride` = {self.config.encoder_stride}."
            )

        if bool_masked_pos is not None:
            size = self.config.image_size // self.config.patch_size
            reshaped_bool_masked_pos = bool_masked_pos.reshape(-1, size, size)
            bool_masked_pixels = (
                reshaped_bool_masked_pos.repeat_interleave(self.config.patch_size, 1)
                .repeat_interleave(self.config.patch_size, 2)
                .unsqueeze(1)
                .contiguous()
            ) # (N, 1, size*patch_size, size*patch_size)
            
            # create new indices 
            indices = torch.randperm(pixel_values.size(0))
            # reorder samples
            reordered_pixel_values = pixel_values[indices].detach() # [N, 3, 224, 224]
    
            # Note: masked part is indicated as 1 in the bool_masked_pixels tensor
            masked_pixel_values = pixel_values * (1 - bool_masked_pixels) + reordered_pixel_values * bool_masked_pixels
        else:
            masked_pixel_values = pixel_values
        
        

        # Obtain vit outputs
        outputs = self.vit(
            masked_pixel_values, #  pixel values masked by noises
            bool_masked_pos=None,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )
        if outputs is None or outputs[0] is None:
            raise ValueError("The ViT model did not return expected outputs.")

        sequence_output = outputs[0]
        sequence_output = sequence_output / sequence_output.norm(p=2, dim=-1, keepdim=True)

        logits = self.classifier(sequence_output[:, 0, :])

        loss = None
        ce_loss = None
        mim_loss = None
        reconstructed_pixel_values = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            ce_loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            loss = ce_loss

            # Calculate MIM loss only when label is available
            # Reshape to (batch_size, num_channels, height, width)
            sequence_output = sequence_output[:, 1:]
            batch_size, sequence_length, num_channels = sequence_output.shape
            height = width = math.floor(sequence_length**0.5)
            sequence_output = sequence_output.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)
    
            # Reconstruct pixel values
            reconstructed_pixel_values = self.mim_decoder(sequence_output)
            
            if bool_masked_pos is not None:
                size = self.config.image_size // self.config.patch_size
                bool_masked_pos = bool_masked_pos.reshape(-1, size, size)
                mask = (
                    bool_masked_pos.repeat_interleave(self.config.patch_size, 1)
                    .repeat_interleave(self.config.patch_size, 2)
                    .unsqueeze(1)
                    .contiguous()
                )
                reconstruction_loss = nn.functional.l1_loss(pixel_values, reconstructed_pixel_values, reduction="none")
                mim_loss = (reconstruction_loss * mask).sum() / (mask.sum() + 1e-5) / self.config.num_channels
                loss += mim_loss


        if not return_dict:
            output = (logits, reconstructed_pixel_values) + outputs[2:]
            return ((loss, mim_loss) + output) if loss is not None else output
        
        output_reconstructed_pixel_values = True
        if output_reconstructed_pixel_values is not None and labels is None:
            sequence_output = sequence_output[:, 1:]
            batch_size, sequence_length, num_channels = sequence_output.shape
            height = width = math.floor(sequence_length**0.5)
            sequence_output = sequence_output.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)
            reconstructed_pixel_values = self.mim_decoder(sequence_output)

        return ImageMultitaskOutput(
            loss=loss,
            logits=logits,
            # ce_loss=ce_loss,
            # mim_loss=mim_loss,
            reconstructed_pixel_values=reconstructed_pixel_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# Note: Config needs to be defined or loaded with the appropriate attributes.
