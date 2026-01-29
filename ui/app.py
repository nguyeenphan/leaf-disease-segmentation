import gradio as gr
from inference import segment_image_multi_model

custom_css = """
.gradio-container {
    font-family: 'Inter', sans-serif;
}


.title {
    text-align: center;
    margin-bottom: 1rem;
}

.description {
    text-align: center;
    color: #666;
    margin-bottom: 2rem;
}

.model-section {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 1.5rem;
    background-color: #f9f9f9;
}

.model-header {
    padding-left: 1rem;
    padding-top: 0.2rem;
    font-size: 1.2em;
    font-weight: 600;
    margin-bottom: 0.2rem;
    color: #333;
}

footer {
    text-align: center;
    margin-top: 2rem;
}
"""

def create_interface():
    """Create and configure Gradio interface"""
    
    with gr.Blocks(css=custom_css, title="Multi-Model Leaf Segmentation") as demo:
        
        # Header
        gr.Markdown(
            """
            # Multi-Model Leaf Segmentation Comparison
            Compare segmentation results from 3 different models simultaneously
            """,
            elem_classes="title"
        )
        
        with gr.Row():
            # Left column - Input (smaller)
            with gr.Column(scale=1):
                gr.Markdown("### Input")
                
                image_input = gr.Image(
                    label="Upload Leaf Image",
                    type="pil",
                    height=400
                )
                
                with gr.Accordion("Advanced Settings", open=False):
                    threshold_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.6,
                        step=0.05,
                        label="Segmentation Threshold",
                        info="Higher values = more strict segmentation"
                    )
                    
                    alpha_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        step=0.05,
                        label="Overlay Transparency",
                        info="0 = transparent, 1 = opaque"
                    )
                    
                    color_dropdown = gr.Dropdown(
                        choices=["green", "red", "blue", "yellow", "cyan", "magenta"],
                        value="green",
                        label="Overlay Color"
                    )
                
                segment_btn = gr.Button(
                    "Run Segmentation",
                    variant="primary",
                    size="lg"
                )
            
            # Right column - Multi-model outputs (larger)
            with gr.Column(scale=3):
                gr.Markdown("### Model Comparison Results")
                
                # Model 1: ResNet18-UNet (Original)
                with gr.Group():
                    gr.Markdown("#### ResNet18-UNet (Original)", elem_classes="model-header")
                    with gr.Row():
                        output_r18_ori_original = gr.Image(label="Original")
                        output_r18_ori_mask = gr.Image(label="Mask")
                        output_r18_ori_overlay = gr.Image(label="Overlay")
                    stats_r18_ori = gr.Markdown(value="", label="Stats")
                
                # Model 2: ResNet18-UNet (Augmented)
                with gr.Group():
                    gr.Markdown("#### ResNet18-UNet (Augmented with Style Transfer)", elem_classes="model-header")
                    with gr.Row():
                        output_r18_aug_original = gr.Image(label="Original")
                        output_r18_aug_mask = gr.Image(label="Mask")
                        output_r18_aug_overlay = gr.Image(label="Overlay")
                    stats_r18_aug = gr.Markdown(value="", label="Stats")
                
                # Model 3: ResNet50-UNet (Augmented)
                with gr.Group():
                    gr.Markdown("#### ResNet50-UNet (Augmented with Style Transfer)", elem_classes="model-header")
                    with gr.Row():
                        output_r50_original = gr.Image(label="Original")
                        output_r50_mask = gr.Image(label="Mask")
                        output_r50_overlay = gr.Image(label="Overlay")
                    stats_r50 = gr.Markdown(value="", label="Stats")
        
        # Overall comparison stats at the bottom
        gr.Markdown("### Overall Comparison Statistics")
        comparison_stats = gr.Markdown(
            value="Upload an image to see comparison statistics.",
            label="Comparison"
        )
        
        # Collect all outputs in order
        all_outputs = [
            # ResNet18-UNet-Ori
            output_r18_ori_original, output_r18_ori_mask, output_r18_ori_overlay,
            # ResNet18-UNet-Aug
            output_r18_aug_original, output_r18_aug_mask, output_r18_aug_overlay,
            # ResNet50-UNet-Aug
            output_r50_original, output_r50_mask, output_r50_overlay,
            # Stats
            stats_r18_ori, stats_r18_aug, stats_r50,
            # Comparison
            comparison_stats
        ]
        
        # Event handlers
        segment_btn.click(
            fn=segment_image_multi_model,
            inputs=[image_input, threshold_slider, alpha_slider, color_dropdown],
            outputs=all_outputs
        )
        
        # Also trigger on image upload for quick results
        image_input.change(
            fn=segment_image_multi_model,
            inputs=[image_input, threshold_slider, alpha_slider, color_dropdown],
            outputs=all_outputs
        )
    
    return demo

def main():
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,        # Default Gradio port
        share=False,             # Set to True to get public URL
        show_error=True,
        inbrowser=True           # Auto-open browser
    )

if __name__ == "__main__":
    main()

