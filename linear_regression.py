import time
import torch
import mlx.core as mx

class MLXImplementation:
    def __init__(self, num_features, num_examples, lr, num_iters):
        self.num_features = num_features
        self.num_examples = num_examples
        self.lr = lr
        self.num_iters = num_iters

    def run(self):
        # Dataset generation
        data_gen_start = time.time()
        w_star = mx.random.normal((self.num_features,))
        X = mx.random.normal((self.num_examples, self.num_features))
        eps = 1e-2 * mx.random.normal((self.num_examples,))
        y = X @ w_star + eps
        data_gen_end = time.time()

        print(f"[MLX] Dataset generation time: {data_gen_end - data_gen_start:.5f} seconds")

        def loss_fn(w):
            return 0.5 * mx.mean(mx.square(X @ w - y))

        grad_fn = mx.grad(loss_fn)
        w = 1e-2 * mx.random.normal((self.num_features,))

        train_start = time.time()
        total_grad_time = 0.0

        for _ in range(10000):
            grad_start = time.time()
            grad = grad_fn(w)
            grad_end = time.time()

            total_grad_time += grad_end - grad_start

            w = w - self.lr * grad
            mx.eval(w)

        train_end = time.time()
        loss = loss_fn(w)
        error_norm = mx.sum(mx.square(w - w_star)).item() ** 0.5

        print(f"[MLX] Training time: {train_end - train_start:.5f} seconds")
        print(f"[MLX] Gradient computation time (total): {total_grad_time:.5f} seconds")
        print(f"[MLX] Loss: {loss.item():.5f}, |w-w*| = {error_norm:.5f}")


class PyTorchImplementation:
    def __init__(self, num_features, num_examples, lr, num_iters):
        self.num_features = num_features
        self.num_examples = num_examples
        self.lr = lr
        self.num_iters = num_iters

    def run(self):
        dtype = torch.float32
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        # Dataset generation
        data_gen_start = time.time()
        w_star = torch.randn(self.num_features, device=device, dtype=dtype)
        X = torch.randn(self.num_examples, self.num_features, device=device, dtype=dtype)
        eps = 1e-2 * torch.randn(self.num_examples, device=device, dtype=dtype)
        y = X @ w_star + eps
        data_gen_end = time.time()

        print(f"[PyTorch] Dataset generation time: {data_gen_end - data_gen_start:.5f} seconds")

        def loss_fn(w):
            return 0.5 * torch.mean((X @ w - y) ** 2)

        w = 1e-2 * torch.randn(self.num_features, device=device, dtype=dtype)

        train_start = time.time()
        total_grad_time = 0.0

        for _ in range(10000):
            grad_start = time.time()

            w.requires_grad = True
            loss = loss_fn(w)
            loss.backward()

            grad_end = time.time()
            total_grad_time += grad_end - grad_start

            with torch.no_grad():
                w -= self.lr * w.grad
                w.grad.zero_()

        train_end = time.time()
        final_loss = loss_fn(w).item()
        error_norm = torch.norm(w - w_star).item()

        print(f"[PyTorch] Training time: {train_end - train_start:.5f} seconds")
        print(f"[PyTorch] Gradient computation time (total): {total_grad_time:.5f} seconds")
        print(f"[PyTorch] Loss: {final_loss:.5f}, |w-w*| = {error_norm:.5f}")


if __name__ == "__main__":
    num_features = 100
    num_examples = 1_000
    num_iters = 10_000
    lr = 0.01

    print("Running MLX implementation...")
    mlx_impl = MLXImplementation(num_features, num_examples, lr, num_iters)
    mlx_impl.run()

    print("\nRunning PyTorch implementation...")
    pytorch_impl = PyTorchImplementation(num_features, num_examples, lr, num_iters)
    pytorch_impl.run()
