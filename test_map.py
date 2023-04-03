from multiprocessing import dummy as multiprocessing

di = []

def run_sh(i):
    print("--",di,i)
    di.append(i)

pool = multiprocessing.Pool(4)
print(range(10))
pool.map(run_sh,range(10))
