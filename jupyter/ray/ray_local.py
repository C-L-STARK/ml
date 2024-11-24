import ray

# init ray once before use.
ray.init()

@ray.remote
def f(x):
    return x * x

result = ray.get([f.remote(i) for i in range(4)])
print(result)