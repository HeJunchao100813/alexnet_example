import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import onnx
import onnxruntime
import os

# 定义数据预处理
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((64, 64)),  # 使用更小的尺寸
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((64, 64)),  # 使用更小的尺寸
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 定义推理辅助函数
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, device, num_epochs=3):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # 每个epoch都有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式
            else:
                model.eval()   # 设置模型为评估模式

            running_loss = 0.0
            running_corrects = 0

            # 迭代数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)  # 将输入移动到GPU
                labels = labels.to(device)  # 将标签移动到GPU

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 仅在训练阶段反向传播和优化
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    return model

def main():
    # 准备数据集目录
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    # 下载CIFAR-10数据集
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=data_transforms['train'])
    val_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=data_transforms['val'])
    dataloaders = {
        'train': torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4),
        'val': torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    }
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
    class_names = train_dataset.classes

    # 加载预训练的AlexNet模型并进行修改
    model = models.alexnet(pretrained=True)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, len(class_names))  # CIFAR-10有10个类别

    # 将模型移动到GPU（如果可用）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 训练模型
    model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, device, num_epochs=3)

    # 导出为ONNX模型
    dummy_input = torch.randn(1, 3, 64, 64, device=device)  # 使用较小尺寸
    onnx_path = "C:\\Users\\HeJunchao\\Desktop\\alexnet.onnx"
    torch.onnx.export(model, dummy_input, onnx_path,
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})
    print(f"Model has been exported to {onnx_path}")

    # 验证ONNX模型
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid.")

    # 使用ONNX Runtime进行推理
    ort_session = onnxruntime.InferenceSession(onnx_path)

    # 测试输入
    x_test = torch.randn(1, 3, 64, 64, device=device)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x_test)}
    ort_outs = ort_session.run(None, ort_inputs)
    print("ONNX model inference completed.")

if __name__ == "__main__":
    main()
