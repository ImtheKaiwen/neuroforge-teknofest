from app.models import RadiologyEncoder, NeuroFusionModel, PathologyEncoder

def train_step(model, optimizer, criterion, mri_batch, slide_batch, labels, device):
    model.train()
    
    # Verileri GPU'ya taşı
    mri_batch = mri_batch.to(device)
    slide_batch = slide_batch.to(device)
    labels = labels.to(device)
    
    optimizer.zero_grad()
    outputs = model(mri_batch, slide_batch)
    loss = criterion(outputs, labels)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()