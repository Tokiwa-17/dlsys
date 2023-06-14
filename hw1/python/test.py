import needle as ndl


if __name__ == '__main__':
    x = ndl.Tensor([1], dtype="float32")
    sum_loss = ndl.Tensor([0], dtype="float32")

    for i in range(100):
        sum_loss = (sum_loss + x * x).detach()

    print(sum_loss.inputs)