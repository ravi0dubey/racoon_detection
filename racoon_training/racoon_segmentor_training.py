from ultralytics import YOLO
import wandb
from wandb.integration.ultralytics import add_wandb_callback

def main():
    # Load the largest YOLOv8 model
    wandb.init(project="racoon_detection",job_type="training")
    model = YOLO('yolov8m.pt')
    # Add W&B Callback for Ultralytics
    add_wandb_callback(model, enable_model_checkpointing=True)
    # Training arguments
    args = {
        'data': './racoon_dataset.yml',
        'epochs': 100,
        'imgsz': 640,
        'batch': 2,
        'device': [0],  # Use GPU 0
        'workers': 16,
        'exist_ok': True,
        'pretrained': True,
        'project': 'racoon_detection',
        'optimizer': 'auto',
        'verbose': True,
        'seed': 0,
        'patience': 50,  # Early stopping patience
        'save': True,
        'save_period': 10,  # Save every 10 epochs
        'cache': 'ram',  # Cache images in RAM for faster training
        'close_mosaic': 70,  # Disable mosaic augmentation for final epochs
        'amp': True,  # Use Automatic Mixed Precision
    }

    # Start training
    results = model.train(**args)
    # Perform validation
    val_results = model.val()
    print(f"Training complete. Results saved to {args['project']}/{args['name']}")
    print(f"Validation results: {val_results}")
    # Perform test
    model(["/home/ravi/racoon_detection/racoon_training/racoon_training_dataset/test/images"])
    # Finalize the W&B Run
    wandb.finish()

if __name__ == '__main__':
    main()

