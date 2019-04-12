import torch
import torchvision.datasets as dataset
import torch.nn as nn
import torchvision.transforms as transforms



model = torch.load('./AlphaX_1.pt', map_location='cpu')

model.eval()

CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]


test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])


test_data = dataset.CIFAR10(root='../data', train=False, download=True, transform=test_transform)
test_queue = torch.utils.data.DataLoader(
        test_data, batch_size=128, shuffle=False, pin_memory=True, num_workers=2)



criterion = nn.CrossEntropyLoss().cuda()

count = 0
sum_loss = 0
sum_acc_top1 = 0
sum_acc_top5 = 0


for step, (data, target) in enumerate(test_queue):
    # target = target.cuda(non_blocking=True)

    with torch.no_grad():
        logits, _ = model(data)
        loss = criterion(logits, target)

        max_k = 5
        __, pred = logits.topk(max_k, 1, True, True)
        matched = pred.t().eq(target.view(1, -1).expand_as(pred.t()))
        acc = []
        for k in (1, max_k):
            matched_k = matched[:k].view(-1).float().sum(0)
            acc.append(matched_k.mul_(100.0 / target.size(0)))

        prec1, prec5 = acc
        n = data.size(0)
        print('Batch', step, 'accuracy is', prec1.item())

        sum_loss += loss.item() * n
        sum_acc_top1 += prec1.item() * n
        sum_acc_top5 += prec5.item() * n
        count += n
        avg_loss = sum_loss / count
        avg_acc_top1 = sum_acc_top1 / count
        avg_acc_top5 = sum_acc_top5 / count


print('Final loss is', avg_loss)
print('Final top_1 test accuracy is', avg_acc_top1)
print('Final top_5 test accuracy is', avg_acc_top5)



