import wandb
from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback
wandb.require("core")
def train():
    # Initialize wandb
    wandb.init(project="racoon_detection",job_type="training")
    # Load the model
    model = YOLO('yolov8s.pt')
    # Add W&B Callback for Ultralytics
    add_wandb_callback(model, enable_model_checkpointing=True)

    # Training arguments
    args = {
        'data': './racoon_dataset.yml',
        'epochs': wandb.config.epochs,
        'imgsz': wandb.config.imgsz,
        'batch': wandb.config.batch,
        'device': [0],
        'workers': 16,
        'project': 'racoon-detection',
        'name': f"rac_det_{wandb.run.id}",
        'exist_ok': True,
        'pretrained': True,
        'optimizer': wandb.config.optimizer,
        'verbose': True,
        'seed': wandb.config.seed,
        'patience': 50,
        'save': True,
        'save_period': 10,
        'cache': 'ram',
        'close_mosaic': 10,
        'amp': True,
        'lr0': wandb.config.lr0,
        'dropout': wandb.config.dropout,
        'scale': wandb.config.scale,
        'erasing': wandb.config.erasing,
    }

    # Start training
    results = model.train(**args)

    # Perform validation
    val_results = model.val()

    print(f"Training complete. Results saved to {args['project']}/{args['name']}")
    print(f"Validation results: {val_results}")

if __name__ == '__main__':
    train()