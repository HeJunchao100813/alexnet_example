import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import onnx
import onnxruntime

# 定义简单的线性回归模型
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # 输入和输出维度都为1

    def forward(self, x):
        return self.linear(x)

# 定义一个函数进行推理
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def main():
    # 1. 准备数据
    # 生成简单的线性数据：y = 2x + 3 + 噪声
    x_train = np.random.rand(100, 1).astype(np.float32)
    y_train = 2 * x_train + 3 + 0.1 * np.random.randn(100, 1).astype(np.float32)

    # 转换为PyTorch的张量格式
    x_train_tensor = torch.from_numpy(x_train)
    y_train_tensor = torch.from_numpy(y_train)

    # 2. 初始化模型、损失函数和优化器
    model = LinearRegressionModel()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 3. 训练模型
    num_epochs = 100
    for epoch in range(num_epochs):
        # 前向传播
        outputs = model(x_train_tensor)
        loss = criterion(outputs, y_train_tensor)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 4. 导出模型为ONNX格式
    dummy_input = torch.tensor([[1.0]])  # 使用单个输入数据作为模型输入的示例
    onnx_path = "C:\\Users\\HeJunchao\\Desktop\\linear_regression.onnx"
    torch.onnx.export(model, dummy_input, onnx_path,
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})

    print(f"Model has been exported to {onnx_path}")

    # 5. 验证ONNX模型
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid.")

    # 6. 使用ONNX Runtime进行推理
    ort_session = onnxruntime.InferenceSession(onnx_path)

    # 测试输入
    x_test = torch.tensor([[4.0]], dtype=torch.float32)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x_test)}
    ort_outs = ort_session.run(None, ort_inputs)

    print(f"Prediction for input 4.0: {ort_outs[0][0][0]}")

if __name__ == "__main__":
    main()
