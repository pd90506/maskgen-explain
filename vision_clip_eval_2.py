import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

from attvis.utils.metrices import batch_pix_accuracy, batch_intersection_union, get_ap_scores, get_f1_scores
from attvis.data.Imagenet import Imagenet_Segmentation
from maskgen.models.random_mask import RandomMaskSaliency
from maskgen.models.vision_maskgen_model9 import MaskGeneratingModel
from captum.attr import IntegratedGradients
from transformers import ViTImageProcessor, ViTForImageClassification, ViTConfig

class ExplanationEvaluator:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_models()
        self.setup_data()
        
    def setup_models(self):
        # Initialize ViT model and processor
        self.processor = ViTImageProcessor.from_pretrained(self.config['model_name'])
        self.pred_model = ViTForImageClassification.from_pretrained(self.config['model_name']).to(self.device)
        self.pred_model.eval()
        
        # Setup attribution methods
        self.setup_attribution_methods()
        
    def setup_attribution_methods(self):
        self.attribution_methods = {
            'ig': IntegratedGradients(
                lambda x: torch.softmax(self.pred_model(pixel_values=x).logits, dim=-1)
            ),
            'rise': RandomMaskSaliency(
                lambda x: self.pred_model(pixel_values=x).logits,
                num_classes=self.config['num_classes']
            ),
            'ours': self._setup_mask_gen_model()
        }
    
    def _setup_mask_gen_model(self):
        config = ViTConfig.from_pretrained(self.config['model_name'])
        mask_gen = MaskGeneratingModel(
            self.pred_model,
            hidden_size=config.hidden_size,
            num_classes=config.num_labels
        ).to(self.device)
        mask_gen.load_state_dict(torch.load(self.config['mask_gen_path']))
        mask_gen.eval()
        return mask_gen
    
    def setup_data(self):
        # Data transforms
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])
        lbl_transform = transforms.Compose([
            transforms.Resize((224, 224), Image.NEAREST),
        ])
        
        # Dataset and dataloader
        dataset = Imagenet_Segmentation(
            self.config['imagenet_seg_path'],
            transform=img_transform,
            target_transform=lbl_transform
        )
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers']
        )
    
    def generate_attribution(self, image, method='ours'):
        """Generate attribution map using specified method"""
        image = image.requires_grad_()
        batch_size = image.shape[0]
        
        if method == 'random':
            return torch.rand(batch_size, 1, 14, 14).to(self.device)
            
        logits = self.pred_model(pixel_values=image).logits
        predicted_class_idx = logits.argmax(-1).item()
        
        if method in self.attribution_methods:
            if method == 'ig':
                attr = self.attribution_methods[method].attribute(
                    image,
                    target=predicted_class_idx,
                    n_steps=200
                ).sum(dim=1)
                return F.avg_pool2d(attr, kernel_size=16, stride=16).reshape(batch_size, 1, 14, 14)
            
            elif method == 'rise':
                return self.attribution_methods[method].attribute_img(
                    image,
                    image_size=self.config['image_size'],
                    patch_size=self.config['patch_size'],
                    n_samples=100,
                    mask_prob=0.5
                ).reshape(batch_size, 1, 14, 14)
            
            elif method == 'ours':
                return self.attribution_methods[method].attribute_img(image).reshape(batch_size, 1, 14, 14)
        
        raise ValueError(f"Unknown attribution method: {method}")
    
    def process_attribution(self, attr_map):
        """Process attribution map for evaluation"""
        # Interpolate to full image size
        attr_map = F.interpolate(attr_map, scale_factor=16, mode='bilinear')
        
        # Normalize
        attr_map = (attr_map - attr_map.min()) / (attr_map.max() - attr_map.min())
        threshold = attr_map.mean()
        
        # Generate binary masks
        mask_fg = attr_map.gt(threshold).float()
        mask_bg = attr_map.le(threshold).float()
        
        return {
            'output': torch.cat((mask_bg, mask_fg), 1),
            'output_ap': torch.cat((1 - attr_map, attr_map), 1),
            'pred': attr_map.clamp(min=self.config['threshold']).div(attr_map.max()).view(-1).cpu().numpy()
        }
    
    def evaluate(self, method='ours'):
        """Run evaluation and return metrics"""
        metrics = {
            'total_correct': 0, 'total_label': 0,
            'total_inter': 0, 'total_union': 0,
            'total_ap': [], 'total_f1': [],
            'predictions': [], 'targets': []
        }
        
        iterator = tqdm(self.dataloader)
        for images, labels in iterator:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Generate and process attribution
            attr_map = self.generate_attribution(images, method)
            processed = self.process_attribution(attr_map)
            
            # Calculate metrics
            batch_metrics = self.calculate_batch_metrics(processed, labels)
            self.update_metrics(metrics, batch_metrics)
            
            # Update progress bar
            current_metrics = self.compute_current_metrics(metrics)
            iterator.set_description(
                f'pixAcc: {current_metrics["pixAcc"]:.4f}, '
                f'mIoU: {current_metrics["mIoU"]:.4f}, '
                f'mAP: {current_metrics["mAP"]:.4f}, '
                f'mF1: {current_metrics["mF1"]:.4f}'
            )
        
        return self.compute_final_metrics(metrics)
    
    def calculate_batch_metrics(self, processed, labels):
        """Calculate metrics for a single batch"""
        output = processed['output'][0]
        correct, labeled = batch_pix_accuracy(output.cpu(), labels[0])
        inter, union = batch_intersection_union(output.cpu(), labels[0], 2)
        ap = np.nan_to_num(get_ap_scores(processed['output_ap'], labels))
        f1 = np.nan_to_num(get_f1_scores(output[1].cpu(), labels[0]))
        
        return {
            'correct': correct.astype('int64'),
            'labeled': labeled.astype('int64'),
            'inter': inter.astype('int64'),
            'union': union.astype('int64'),
            'ap': ap,
            'f1': f1,
            'pred': processed['pred'],
            'target': labels.view(-1).cpu().numpy()
        }
    
    def compute_final_metrics(self, metrics):
        """Compute final evaluation metrics"""
        predictions = np.concatenate(metrics['predictions'])
        targets = np.concatenate(metrics['targets'])
        precision, recall, _ = precision_recall_curve(targets, predictions)
        
        pixAcc = np.float64(1.0) * metrics['total_correct'] / (np.spacing(1) + metrics['total_label'])
        IoU = np.float64(1.0) * metrics['total_inter'] / (np.spacing(1) + metrics['total_union'])
        
        return {
            'pixel_accuracy': pixAcc * 100,
            'mean_iou': IoU.mean(),
            'mean_ap': np.mean(metrics['total_ap']),
            'mean_f1': np.mean(metrics['total_f1']),
            'precision': precision,
            'recall': recall
        }

# Example configuration
config = {
    'model_name': 'google/vit-base-patch16-224',
    'mask_gen_path': 'mask_gen_model/mask_gen_model_1_150.pth',
    'imagenet_seg_path': 'data/gtsegs_ijcv.mat',
    'batch_size': 1,
    'num_workers': 0,
    'num_classes': 1000,
    'image_size': 224,
    'patch_size': 16,
    'threshold': 0.0
}

# Example usage
if __name__ == "__main__":
    evaluator = ExplanationEvaluator(config)
    results = evaluator.evaluate(method='ours')
    
    print(f"Pixel-wise Accuracy: {results['pixel_accuracy']:.2f}%")
    print(f"Mean IoU: {results['mean_iou']:.4f}")
    print(f"Mean AP: {results['mean_ap']:.4f}")
    print(f"Mean F1: {results['mean_f1']:.4f}")
    
    # Plot PR curve
    plt.figure()
    plt.plot(results['recall'], results['precision'])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig('PR_curve.png')