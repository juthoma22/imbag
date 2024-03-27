import torch
import wandb
from tqdm import tqdm
from transformers import CLIPModel

class CLIPFineTuner:
    def __init__(self, model_name="openai/clip-vit-large-patch14-336", batch_size=8, learning_rate=5e-6, epochs=3, argument="all"):
        self.model_name = model_name
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.argument = argument

        print("Loading model...")
        # Set up model and optimizer
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        else:
            self.device = torch.device("cpu")
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch, return_loss=True)
                loss = outputs.loss
                total_loss += loss.item()

                logits_per_image = outputs.logits_per_image
                predictions = logits_per_image.argmax(dim=-1)
                correct = (predictions == torch.arange(len(predictions), device=self.device)).sum()
                total_correct += correct.item()
                total_samples += len(predictions)

        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_samples
        return avg_loss, accuracy

    def train(self, train_loader, validation_loader):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(self.epochs):
            self.model.train()
            total_train_loss = 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch, return_loss=True)
                loss = outputs.loss

                total_train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            avg_train_loss = total_train_loss / len(train_loader)
            avg_val_loss, val_accuracy = self.evaluate(validation_loader)

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_accuracy": val_accuracy
            })
            print(f"Epoch {epoch+1} - Avg Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # run id from wandb
        run_id = wandb.run.name
        path = f'models/{run_id}_{epoch}.pth'
        self.save_model(path)

    def save_model(self, path):
        self.model.to('cpu')
        torch.save(self.model.state_dict(), path)