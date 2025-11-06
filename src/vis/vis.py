import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from transformers import AutoProcessor, LlavaForConditionalGeneration

class AttentionVisualizer:
    def __init__(self, model_name="llava-hf/llava-1.5-7b-hf"):
        print(f"Loading model: {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            attn_implementation="eager"
        ).to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")

    def get_cross_modal_attention(self, image, text_query, layer_idx=14):
        prompt = f"USER: <image>\n{text_query}\nASSISTANT:"
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)

    
        with torch.no_grad():
            model_outputs = self.model(
                **inputs,
                output_attentions=True,
                use_cache=False
            )

 
        attentions = model_outputs.attentions[layer_idx]  # [batch, heads, seq, seq]
        

        avg_attn = attentions.mean(dim=1)[0]  # [seq_len, seq_len]
        

        last_token_attn = avg_attn[-1].cpu().numpy()
        
        input_ids = inputs['input_ids'][0].cpu().numpy()
        seq_len = last_token_attn.shape[0]
        try:
            image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<image>")
            image_positions = np.where(input_ids == image_token_id)[0]
            
            if len(image_positions) > 0:

                image_start = image_positions[0] + 1
                num_image_tokens = 576
            else:
                image_start = 1
                num_image_tokens = 576
        except:
            image_start = 1
            num_image_tokens = 576
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False
            )
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        return last_token_attn, response, image_start, num_image_tokens

    def create_attention_heatmap(self, image, attention_weights, image_start, num_image_tokens):
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        if num_image_tokens <= 0:
            print("Warning: No image tokens found!")
            return img_array, np.zeros((h, w))
        

        image_attention = attention_weights[image_start:image_start + num_image_tokens]
        
        print(f"Image attention shape: {image_attention.shape}")
        print(f"Image attention range: [{image_attention.min():.6f}, {image_attention.max():.6f}]")
        

        if len(image_attention) != num_image_tokens:
            print(f"Warning: Expected {num_image_tokens} tokens, got {len(image_attention)}")
            num_image_tokens = len(image_attention)
        

        if image_attention.max() - image_attention.min() > 1e-8:
            image_attention = (image_attention - image_attention.min()) / (image_attention.max() - image_attention.min())
        else:
            print("Warning: Attention values are constant!")
        

        grid_size = int(np.sqrt(num_image_tokens))
        
        if grid_size * grid_size == num_image_tokens:
            attention_map = image_attention.reshape(grid_size, grid_size)
        else:
            grid_size = int(np.floor(np.sqrt(num_image_tokens)))
            attention_map = image_attention[:grid_size*grid_size].reshape(grid_size, grid_size)
            print(f"Warning: Non-square attention map, using {grid_size}x{grid_size}")
        
        print(f"Attention map shape after reshape: {attention_map.shape}")
        

        if attention_map.ndim != 2:
            print(f"Error: Attention map has wrong dimensions: {attention_map.ndim}")
            return img_array, np.zeros((h, w))
        
        attention_map = attention_map.astype(np.float32)
        
        # Resize to match image dimensions
        try:
            attention_resized = cv2.resize(attention_map, (w, h), interpolation=cv2.INTER_LINEAR)
        except cv2.error as e:
            print(f"OpenCV resize error: {e}")
            print(f"Attention map dtype: {attention_map.dtype}, shape: {attention_map.shape}")
            print(f"Target size: ({w}, {h})")
            # Fallback: use numpy
            from scipy.ndimage import zoom
            zoom_factors = (h / attention_map.shape[0], w / attention_map.shape[1])
            attention_resized = zoom(attention_map, zoom_factors, order=1)
        
        return img_array, attention_resized

    def generate_fig(self, image_path, output_path="figure1_reproduction.png", layer_idx=14):
        
        image = Image.open(image_path).convert('RGB')
        
        # Queries from the paper
        query_en = "Is there a bird in the image?"
        query_zh = "图像中有鸟吗？"
        
        print(f"\nProcessing English query at layer {layer_idx}...")
        attn_en, resp_en, start_en, num_en = self.get_cross_modal_attention(
            image, query_en, layer_idx
        )
        print(f"English response: {resp_en}")
        print(f"English - image_start: {start_en}, num_tokens: {num_en}")
        
        print(f"\nProcessing Chinese query at layer {layer_idx}...")
        attn_zh, resp_zh, start_zh, num_zh = self.get_cross_modal_attention(
            image, query_zh, layer_idx
        )
        print(f"Chinese response: {resp_zh}")
        print(f"Chinese - image_start: {start_zh}, num_tokens: {num_zh}")
        
        # Create heatmaps
        print("\nCreating English heatmap...")
        img_en, heatmap_en = self.create_attention_heatmap(image, attn_en, start_en, num_en)
        
        print("\nCreating Chinese heatmap...")
        img_zh, heatmap_zh = self.create_attention_heatmap(image, attn_zh, start_zh, num_zh)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6)) 
        

        ax_en = axes[0]
        ax_en.imshow(img_en)
        ax_en.imshow(heatmap_en, cmap='jet', alpha=0.6, vmin=0, vmax=1)
        ax_en.axis('off')
        ax_en.set_title('English Query\nAttention Map', fontsize=12, fontweight='bold')
        
       
        ax_zh = axes[1]
        ax_zh.imshow(img_zh)
        ax_zh.imshow(heatmap_zh, cmap='jet', alpha=0.6, vmin=0, vmax=1)
        ax_zh.axis('off')
        ax_zh.set_title('Chinese Query\nAttention Map', fontsize=12, fontweight='bold')
        
        plt.tight_layout()

        
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\n✓ Figure saved to: {output_path}")
        plt.close()
        
        return resp_en, resp_zh


def main():
    visualizer = AttentionVisualizer()
    
    image_path = "/data/asca/MMTQA/src/vis/download.png"
    
    #
    for layer in range(0,31):
        print(f"\n{'='*60}")
        print(f"Processing Layer {layer}")
        print(f"{'='*60}")
        
        try:
            resp_en, resp_zh = visualizer.generate_fig(
                image_path, 
                output_path=f"layer{layer}.png",
                layer_idx=layer
            )
            
            print(f"\n✓ Results for Layer {layer}:")
            print(f"  English: {resp_en.split('ASSISTANT:')[-1].strip()[:100]}")
            print(f"  Chinese: {resp_zh.split('ASSISTANT:')[-1].strip()[:100]}")
        except Exception as e:
            print(f"\n✗ Error processing layer {layer}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()