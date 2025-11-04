import sys
import tempfile
from pathlib import Path
from utils.report_generator import ReportGenerator

# 模拟配置
test_config = {
    "experiment": {
        "task": "cifar10",
        "device": "cuda",
        "seed": 42
    },
    "model": {
        "arch": "resnet18_cifar",
        "num_classes": 10
    },
    "optimizer": {
        "name": "F3EO",
        "lr": 0.1,
        "weight_decay": 5e-4
    },
    "data": {
        "batch_size": 128
    }
}

# 模拟训练数据
def generate_test_data():
    import random
    import time
    
    # 创建临时输出目录
    output_dir = Path(tempfile.mkdtemp()) / "test_experiment"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化报告生成器
    report_gen = ReportGenerator(test_config, output_dir)
    
    # 模拟5个epoch的训练数据
    for epoch in range(5):
        # 模拟训练结果（损失递减，准确率递增）
        train_loss = 2.0 - epoch * 0.3 + random.uniform(-0.1, 0.1)
        valid_loss = 2.2 - epoch * 0.25 + random.uniform(-0.1, 0.1)
        train_acc = 50.0 + epoch * 8.0 + random.uniform(-2, 2)
        valid_acc = 48.0 + epoch * 7.5 + random.uniform(-2, 2)
        
        train_results = {"loss": max(0.1, train_loss), "accuracy": min(99.0, train_acc)}
        valid_results = {"loss": max(0.1, valid_loss), "accuracy": min(99.0, valid_acc)}
        
        lr = 0.1 * (0.9 ** epoch)
        epoch_time = 45.0 + random.uniform(-5, 5)
        
        report_gen.log_epoch(epoch, train_results, valid_results, lr, epoch_time)
        
        print(f"Epoch {epoch+1}: Train Acc: {train_results['accuracy']:.2f}%, Valid Acc: {valid_results['accuracy']:.2f}%")
    
    # 生成报告
    report = report_gen.generate_summary()
    
    print(f"\n{'='*60}")
    print("MARKDOWN REPORT GENERATED:")
    print(f"{'='*60}")
    print(report)
    print(f"{'='*60}")
    print(f"\nReport saved to: {output_dir}/summary.md")
    
    return output_dir

if __name__ == "__main__":
    output_dir = generate_test_data()
    print(f"\nTest completed. Report location: {output_dir}")