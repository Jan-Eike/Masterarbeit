from concurrent.futures import ThreadPoolExecutor
import threading
import time


def task(n):
    print('{}: sleeping {}'.format(
        threading.current_thread().name,
        n)
    )
    time.sleep(n / 10)
    print('{}: done with {}'.format(
        threading.current_thread().name,
        n)
    )
    return n / 10


print('main: starting')
all = []
with ThreadPoolExecutor(max_workers=20) as executor:
    for i in range(5, 0, -1):
        future = executor.submit(task, i)
        all.append(future.result())
print(all)
print('main: future: {}'.format(future))
print('main: waiting for results')
result = future.result()
print('main: result: {}'.format(result))
print('main: future after result: {}'.format(future))